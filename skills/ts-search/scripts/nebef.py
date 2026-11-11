#!/usr/bin/env python3
"""Print the force, energy and relative energy of NEB images.
Usage:
    # cd to the results of NEB
    python3 nebef.py -o result.txt
"""

import argparse
import bz2
import gzip
import os
import re
import sys


IMAGE_RE = re.compile(r"^\d\d$")


def open_text(path):
    """Open a (possibly gzip/bzip2 compressed) text file."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt")
    return open(path)


def outcar_file(image_dir):
    """Return the OUTCAR of an image (plain or compressed), if present."""
    for name in ("OUTCAR", "OUTCAR.gz", "OUTCAR.bz2"):
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return path
    return None


def image_dirs(root="."):
    """Sorted list of image directories that contain an OUTCAR."""
    dirs = [
        d
        for d in os.listdir(root)
        if IMAGE_RE.match(d)
        and os.path.isdir(os.path.join(root, d))
        and outcar_file(os.path.join(root, d)) is not None
    ]
    dirs.sort()
    return dirs


def last_energy_line(path):
    """Last ``energy  without`` line, or ``None``."""
    line = None
    with open_text(path) as fh:
        for text in fh:
            if "energy  without" in text:
                line = text
    return line


def last_force_line(path):
    """Last line containing ``max at`` (the FORCES line), or ``None``."""
    line = None
    with open_text(path) as fh:
        for text in fh:
            if "max at" in text:
                line = text
    return line


def nebef_table(root="."):
    """Yield ``(index, force, energy, relative_energy)`` per image."""
    energy = None
    for i, image in enumerate(image_dirs(root)):
        path = outcar_file(os.path.join(root, image))

        energy_line = last_energy_line(path)
        force_line = last_force_line(path)
        if energy_line is None or force_line is None:
            continue

        # Same character columns as the perl ``cut -c`` calls.
        value = float(energy_line[66:78])
        force = float(force_line[26:38])

        if energy is None:
            energy = value
        yield i, force, value, value - energy


def render(rows):
    return "".join(
        "%4i %16.6f %16.6f %16.6f \n" % row for row in rows
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the force/energy table of a NEB run "
        "(an ASE-free reproduction of nebef.pl)."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the table to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    table = render(nebef_table("."))

    if args.output is None:
        sys.stdout.write(table)
    else:
        with open(args.output, "w") as fh:
            fh.write(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
