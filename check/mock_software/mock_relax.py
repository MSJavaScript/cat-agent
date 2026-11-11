#!/usr/bin/env python3
"""Fake 'mock-relax' software.

Reads one input JSON, sleeps for a moment, and writes a deterministic
'relaxed' JSON plus a log.  It is deliberately simple so the whole agent test
is fast, but it still runs as a real jobmanager job.
"""
import argparse
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fake structure relaxation.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--seconds", type=float, default=0.8, help="Fake run time.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    time.sleep(max(0.0, args.seconds))

    # Deterministic pseudo-energy: lower for sample A than for sample B.
    sample = str(data.get("sample", "X"))
    energy = 10.0 + float(data.get("seed", 0)) * 0.1 + float(data.get("scale", 1.0))
    if sample == "B":
        energy += 0.75

    data["energy"] = round(energy, 6)
    data["relaxed"] = True
    data["software"] = "mock-relax 1.0"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # The skill documents ``relax_<sample>.log`` (for example relax_A.log),
    # not a log named after the --output stem.
    log_name = f"relax_{sample}.log"
    log_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), log_name)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"mock-relax finished for sample {sample}\n")
        f.write(f"energy = {data['energy']}\n")

    print(f"relaxed sample {sample}: energy={data['energy']} -> {args.output}")


if __name__ == "__main__":
    main()
