#!/usr/bin/env python3
"""Fake 'mock-prepare' software used by check/test_agent_e2e.py.

It creates two tiny JSON inputs.  The sleep is intentional: it makes even the
quick test exercises real wall-clock waiting and async concurrency.
"""
import argparse
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mock inputs for samples A and B.")
    parser.add_argument("--outdir", required=True, help="Directory for input_A.json and input_B.json.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for index, sample in enumerate(("A", "B")):
        payload = {
            "sample": sample,
            "seed": args.seed + index,
            "scale": 1.0 + 0.5 * index,
            "note": "Synthetic input. No real physics is performed.",
        }
        path = os.path.join(args.outdir, f"input_{sample}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    time.sleep(0.3)
    print(f"Created input_A.json and input_B.json in {args.outdir}")


if __name__ == "__main__":
    main()
