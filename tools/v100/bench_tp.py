#!/usr/bin/env python3
"""Run the saved code-task occupancy/capacity matrix through the public NInfer benchmark.

One process/model at a time. Saved prompts must contain their final task and assistant suffix;
capacity is never substituted for occupied tokens. The output is a fixed-window performance
measurement, not a code-quality score. No external engine or model conversion is performed here.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

from compare_llama import EOG_TOKENS


def summarize(report):
    test, = report["tests"]
    wall, phase, prefill, accepted, drafted = [], [], [], 0, 0
    outputs, speculative = [], []
    for rep in test["reps"]:
        ids = rep["generation"]["token_ids"]
        count = test["n_gen"]
        if (rep["decode_output_tokens"] != count or len(ids) != count + 1
                or rep["generated_output_tokens"] != count + 1
                or rep["generation"]["finish_reason"] != "output_limit"
                or EOG_TOKENS.intersection(ids)):
            raise RuntimeError("invalid fixed-window generation: inspect the raw report")
        times = rep["timings"]
        wall.append(count / (times["total_seconds"] - times["first_token_seconds"]))
        phase.append(count / times["decode_seconds"])
        prefill.append(test["n_prompt"] / times["prefill_seconds"])
        accepted += rep["speculative"]["accepted_tokens"]
        drafted += rep["speculative"]["drafted_tokens"]
        outputs.append(ids)
        speculative.append(rep["speculative"])
    def stats(values):
        return {"mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0}
    return {"prompt_tokens": test["n_prompt"], "timed_decode_tokens": test["n_gen"],
            "capacity": report["config"]["max_context"],
            "allocated_kv_capacity": report["memory"]["kv_capacity"],
            "prefill_tok_s": stats(prefill), "decode_phase_tok_s": stats(phase),
            "decode_wall_tok_s": stats(wall), "accepted_tokens": accepted,
            "drafted_tokens": drafted, "acceptance_rate": accepted / drafted if drafted else 0,
            "repetition_ids_match": all(ids == outputs[0] for ids in outputs),
            "repetition_speculative_match": all(stats == speculative[0] for stats in speculative)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=Path("build-v100/bench/ninfer_bench"))
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-chunk", type=int, default=2560)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--suite", choices=("occupancy", "capacity", "plain"), required=True)
    parser.add_argument("--capacity", type=int, default=180000,
                        help="KV capacity for the occupancy/plain suites")
    args = parser.parse_args()
    if len(args.devices.split(",")) != args.tp or args.repetitions < 1:
        parser.error("device count must equal TP width and repetitions must be positive")
    if args.suite == "occupancy":
        cases = [(args.capacity, n, 3) for n in (3072, 8192, 16384, 32768, 65536, 85000)
                 if n + 1024 <= args.capacity]
    elif args.suite == "capacity":
        cases = [(n, 512, 3) for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536)]
    else:
        cases = [(args.capacity, n, 0) for n in (3072, 85000) if n + 1024 <= args.capacity]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # 512 prompt + 1 prefill output + 512 decode would exceed the 1024-capacity row.
    # Use the same 256-token decode window at every capacity; occupancy/plain use 512.
    decode_tokens = 256 if args.suite == "capacity" else 512
    for capacity, tokens, drafts in cases:
        name = f"tp{args.tp}-{args.suite}-ctx{capacity}-pp{tokens}-mtp{drafts}"
        raw = args.output_dir / (name + ".json")
        corpus = args.corpus_dir / ("v100-code-85000-iommu-pt.ids" if tokens == 85000
                                    else f"code-{tokens}.ids")
        command = [str(args.bench.resolve()), "--weights", str(args.weights.resolve()),
                   "--tp", str(args.tp), "--devices", args.devices, "--max-ctx", str(capacity),
                   "--kv-dtype", "int8", "--prefill-chunk", str(args.prefill_chunk),
                   "--spec", "mtp" if drafts else "none", "--draft-tokens", str(drafts),
                   *(["--lm-head-draft"] if drafts else []), "--corpus", str(corpus.resolve()),
                   "-pg", f"{tokens},{decode_tokens}", "--warmup", "0", "-r", str(args.repetitions),
                   "--capture-generation", "-o", "json", "--output-file", str(raw.resolve())]
        print("Running:", name, flush=True)
        (args.output_dir / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
        with (args.output_dir / (name + ".log")).open("w") as log:
            subprocess.run(command, env=os.environ.copy(), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        report = json.loads(raw.read_text())
        row = summarize(report)
        row.update(tp=args.tp, devices=args.devices, drafts=drafts, raw_report=raw.name)
        rows.append(row)
        (args.output_dir / f"{args.suite}-summary.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
