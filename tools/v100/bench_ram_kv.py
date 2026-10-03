#!/usr/bin/env python3
"""Native NInfer RAM-KV fixture evaluation, under a whole-server 32 GB cgroup cap.

Uses existing fixed fixtures; never generates synthetic throughput token streams or downloads
weights. Nine factual queries + a deliberately bounded 512-token code window, not code success.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from health_log import HealthLog, durable_json


def http(url, payload=None):
    request = urllib.request.Request(url, None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2400) as response:
        return json.load(response)


def score(case, text):
    expected = case["expected"]
    if isinstance(expected, str):
        return {"correct": text.strip().strip('`"') == expected,
                "expected_present": expected in text}
    if case["kind"] == "json":
        try:
            actual = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        except (ValueError, TypeError):
            actual = {}
        fields = sum(actual.get(k) == v for k, v in expected.items()) if isinstance(actual, dict) else 0
        return {"correct": fields == len(expected), "fields_correct": fields, "fields": len(expected)}
    return {"correct": None, "credentials_present": sum(v in text for v in expected.values()),
            "credentials": len(expected), "code_window_only": True}


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--draft-tokens", type=int, choices=(0, 3), default=3)
    parser.add_argument("--port", type=int, default=18381)
    parser.add_argument("--case", action="append", help="optional case names; default all ten")
    parser.add_argument("--no-prefix-reuse", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    request_log = args.output.resolve() / "requests.jsonl"
    if request_log.exists():
        parser.error("use a new output directory; request log already exists")
    command = [str(args.server.resolve()), str(args.weights.resolve()), "--tp", "2", "--devices", "0,1",
               "--max-context", str(fixture["capacity"]), "--prefill-chunk", "1024", "--kv-dtype", "int8",
               "--host", "127.0.0.1", "--port", str(args.port), "--greedy", "--no-thinking",
               "--log-stats-interval-ms", "1000", "--request-log-jsonl", str(request_log)]
    if args.window:
        command += ["--ram-kv-window", str(args.window), "--ram-kv-budget-bytes", "32000000000"]
    if args.draft_tokens:
        command += ["--spec", "mtp", "--draft-tokens", str(args.draft_tokens), "--lm-head-draft"]
    if args.no_prefix_reuse:
        command += ["--no-prefix-reuse"]
    unit = f"ninfer-ram-kv-{os.getpid()}.scope"
    launch = ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit,
              "-p", "MemoryMax=32000000000", "-p", "MemorySwapMax=0", *command]
    report = {"engine": "NInfer public Engine / native RAM-KV", "weights": str(args.weights),
              "capacity": fixture["capacity"], "window": args.window,
              "approximate": bool(args.window and args.window < fixture["capacity"]),
              "policy": "lexical-256, sink512, recent4096" if args.window else "full history",
              "command": command, "whole_process_memory_limit_bytes": 32000000000,
              "draft_tokens": args.draft_tokens, "cases": [], "status": "running",
              "health_log": str(args.output.resolve() / "health"),
              "diagnostic_logging": {"gpu_interval_ms": 200, "host_interval_ms": 1000,
                                     "engine_stats_interval_ms": 1000}}
    durable_json(args.output / "report.json", report)
    url = f"http://127.0.0.1:{args.port}"
    with HealthLog(args.output / "health") as health, (args.output / "server.log").open("w") as log:
        health.follow_file(args.output / "server.log", "ninfer-console.log")
        health.event("server_launch", command=command, scope=unit)
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see server.log")
                try:
                    models = http(url + "/v1/models")
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("server startup timeout")
                    time.sleep(0.5)
            cgroup = subprocess.check_output(["systemctl", "--user", "show", unit,
                                              "-p", "ControlGroup", "--value"], text=True).strip()
            memory_peak = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.peak"
            health.attach_cgroup(memory_peak.parent)
            health.follow_file(request_log, "ninfer-requests.jsonl")
            model = models["data"][0]["id"]
            for case in fixture["cases"]:
                if args.case and case["name"] not in args.case:
                    continue
                report["active_case"] = case["name"]
                durable_json(args.output / "report.json", report)
                health.event("case_start", name=case["name"], capacity=fixture["capacity"], window=args.window)
                start = time.monotonic()
                response = http(url + "/v1/chat/completions", {"model": model,
                    "messages": case["messages"], "max_tokens": case["limit"],
                    "temperature": 0, "seed": 13, "presence_penalty": 0, "frequency_penalty": 0,
                    "stream": False})
                elapsed = time.monotonic() - start
                text = response["choices"][0]["message"].get("content") or ""
                done = [r for r in records(request_log) if r.get("event") == "request_done"][-1]
                result, timings = done["result"], done["timings_seconds"]
                row = {"name": case["name"], "usage": response["usage"], "result": result,
                       "timings_seconds": timings, "http_seconds": elapsed,
                       "speculative": done["speculative"], "quality": score(case, text), "output": text,
                       "prefill_tok_s": result["computed_prefill_tokens"] / timings["prefill"] if timings["prefill"] else 0,
                       "committed_decode_tok_s": (result["completion_tokens"] - 1) / timings["decode"] if timings["decode"] else 0,
                       "memory_peak_bytes": int(memory_peak.read_text())}
                report["cases"].append(row)
                report["active_case"] = None
                durable_json(args.output / "report.json", report)
                health.event("case_done", name=case["name"], prompt_tokens=result["prompt_tokens"],
                             http_seconds=elapsed, quality=row["quality"])
                print(f"{case['name']}: input={result['prompt_tokens']} reused={result['prefix_cache_hit_tokens']} "
                      f"decode={row['committed_decode_tok_s']:.2f} tok/s quality={row['quality']} "
                      f"peak={row['memory_peak_bytes']/1e9:.2f} GB", flush=True)
            report["startup"] = [r for r in records(request_log) if r.get("event") == "server_start"]
            report["memory_peak_bytes"] = int(memory_peak.read_text())
            report["status"] = "complete"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = repr(error)
            health.event("benchmark_failed", error=repr(error))
            raise
        finally:
            durable_json(args.output / "report.json", report)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)


if __name__ == "__main__":
    main()
