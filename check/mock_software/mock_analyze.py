#!/usr/bin/env python3
"""Fake 'mock-analyze' software.

Collects all relaxed_*.json files in a directory, compares their fake
energies, and writes analysis.json plus analysis.md.
"""
import argparse
import glob
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fake relaxation results.")
    parser.add_argument("--input-dir", required=True, help="Directory containing relaxed_*.json.")
    parser.add_argument("--output", required=True, help="Output analysis.json path.")
    parser.add_argument("--seconds", type=float, default=0.4, help="Fake analysis time.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "relaxed_*.json")))
    if not paths:
        raise SystemExit(f"No relaxed_*.json found in {args.input_dir}")

    results = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.append({
            "sample": data.get("sample", os.path.basename(path)),
            "energy": float(data["energy"]),
            "source": os.path.basename(path),
        })

    time.sleep(max(0.0, args.seconds))

    best = min(results, key=lambda item: item["energy"])
    summary = {
        "software": "mock-analyze 1.0",
        "results": results,
        "best_sample": best["sample"],
        "best_energy": best["energy"],
        "energy_gap": round(max(r["energy"] for r in results) - min(r["energy"] for r in results), 6),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Mock analysis\n\n")
        f.write("| sample | energy |\n|---|---|\n")
        for item in sorted(results, key=lambda r: r["sample"]):
            f.write(f"| {item['sample']} | {item['energy']} |\n")
        f.write(f"\nLowest energy: sample {best['sample']} ({best['energy']}).\n")

    print(f"Compared {len(results)} samples; best={best['sample']} energy={best['energy']}")


if __name__ == "__main__":
    main()
