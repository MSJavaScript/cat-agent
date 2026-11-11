#!/usr/bin/env python3
"""Fake 'mock-report' software: turn analysis.json into a markdown report."""
import argparse
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown report from mock analysis.")
    parser.add_argument("--analysis", required=True, help="analysis.json path.")
    parser.add_argument("--output", required=True, help="Output markdown path.")
    args = parser.parse_args()

    with open(args.analysis, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    time.sleep(0.2)

    lines = [
        "# Mock compute report",
        "",
        f"Best sample: **{analysis['best_sample']}**",
        f"Best energy: {analysis['best_energy']}",
        f"Energy gap: {analysis['energy_gap']}",
        "",
        "| sample | energy |",
        "|---|---|",
    ]
    for item in sorted(analysis["results"], key=lambda r: r["sample"]):
        lines.append(f"| {item['sample']} | {item['energy']} |")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
