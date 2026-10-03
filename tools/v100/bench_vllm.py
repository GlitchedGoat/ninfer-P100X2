#!/usr/bin/env python3
"""Measure a running 1Cat/vLLM TP4 control using the saved NInfer prompt IDs.

The server's capacity, KV format and MTP window must be recorded separately.
This measures HTTP wall time, not GPU phase time. Unique cache salts isolate cold
requests; returned token IDs make speculative multi-token SSE chunks countable.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request
import uuid

from compare_llama import EOG_TOKENS


def measure(base_url, model, prompt, output_tokens):
    payload = {"model": model, "prompt": prompt, "max_tokens": output_tokens,
               "temperature": 0, "top_p": 1, "ignore_eos": True,
               "stream": True, "stream_options": {"include_usage": True},
               "return_token_ids": True, "cache_salt": uuid.uuid4().hex}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first = last = None
    first_count = 0
    ids, text = [], []
    usage = finish = None
    done = False
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            item = json.loads(data)
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                chunk = choice.get("token_ids") or []
                if chunk:
                    now = time.perf_counter()
                    if first is None:
                        first, first_count = now, len(chunk)
                    last = now
                    ids.extend(chunk)
                text.append(choice.get("text") or "")
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    total = time.perf_counter() - start
    if (not done or not usage or usage.get("prompt_tokens") != len(prompt)
            or usage.get("completion_tokens") != output_tokens
            or len(ids) != output_tokens or finish != "length"
            or first is None or last <= first):
        raise RuntimeError("incomplete or uncountable fixed-window HTTP generation")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if cached:
        raise RuntimeError(f"cold benchmark unexpectedly reused {cached} prompt tokens")
    return {"prompt_tokens": len(prompt), "output_tokens": len(ids),
            "first_chunk_tokens": first_count, "ttft_seconds": first - start,
            "decode_wall_seconds": last - first, "total_seconds": total,
            "decode_wall_tok_s": (len(ids) - first_count) / (last - first),
            "eog_tokens": len(EOG_TOKENS.intersection(ids)), "usage": usage,
            "token_ids": ids, "text": "".join(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capacity", type=int, required=True,
                        help="actual startup max-model-len; not a request override")
    parser.add_argument("--inputs", default="3072,8192,16384,32768,65536,85000")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=513)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for n in map(int, args.inputs.split(",")):
        path = args.corpus_dir / ("v100-code-85000-iommu-pt.ids" if n == 85000
                                  else f"code-{n}.ids")
        prompt = [int(word) for word in path.read_text().split()]
        if len(prompt) != n or n + args.output_tokens > args.capacity:
            raise RuntimeError(f"invalid saved prompt or insufficient capacity: {path}")
        reps = []
        for i in range(args.repetitions):
            print(f"Running HTTP pp{n}, repetition {i + 1}", flush=True)
            rep = measure(args.base_url, args.model, prompt, args.output_tokens)
            reps.append(rep)
            (args.output_dir / f"pp{n}-rep{i + 1}.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
        def stats(key):
            values = [rep[key] for rep in reps]
            return {"mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0}
        row = {"prompt_tokens": n, "capacity": args.capacity,
               "output_tokens": args.output_tokens,
               "ttft_seconds": stats("ttft_seconds"),
               "decode_wall_tok_s": stats("decode_wall_tok_s"),
               "repetition_ids_match": all(r["token_ids"] == reps[0]["token_ids"] for r in reps),
               "eog_tokens": sum(r["eog_tokens"] for r in reps)}
        summary.append(row)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
