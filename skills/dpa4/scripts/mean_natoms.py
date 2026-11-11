#!/usr/bin/env python3
"""Calculate the mean number of atoms per structure in a DeePMD HDF5 file.
Usage:
    python3 mean_natoms.py --hdf5 data.hdf5
    python3 mean_natoms.py --hdf5 data.hdf5 --nsample 100
"""

from __future__ import annotations

import argparse
import random

import h5py


def count_frames(group: h5py.Group) -> int:
    """Number of frames stored in a system group (sum over set.* folders)."""
    nframes = 0
    for key in group:
        sub = group[key]
        if key.startswith("set.") and isinstance(sub, h5py.Group):
            if "coord.npy" in sub:
                nframes += sub["coord.npy"].shape[0]
    return nframes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the mean number of atoms of structures in a DeePMD HDF5 file."
    )
    parser.add_argument(
        "--hdf5",
        required=True,
        help="path to the HDF5 file (deepmd/hdf5 format)",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=None,
        help="randomly select this many structures; if omitted, use all structures",
    )
    args = parser.parse_args()

    # number of atoms of every structure (frame) in the file
    natoms_per_structure: list[int] = []
    with h5py.File(args.hdf5, "r") as f:
        for name in f:
            group = f[name]
            if not isinstance(group, h5py.Group) or "type.raw" not in group:
                continue
            natoms = group["type.raw"].shape[0]
            natoms_per_structure.extend([natoms] * count_frames(group))

    total = len(natoms_per_structure)
    if total == 0:
        raise SystemExit(f"no structures found in {args.hdf5}")

    sampled = natoms_per_structure
    if args.nsample is not None:
        if args.nsample < 1:
            raise SystemExit("--nsample must be a positive integer")
        # clamp to the total number of structures
        sampled = random.sample(natoms_per_structure, min(args.nsample, total))

    mean_natoms = sum(sampled) / len(sampled)
    print(f"structures used      : {len(sampled)} / {total}")
    print(f"mean number of atoms : {mean_natoms:.4f}")


if __name__ == "__main__":
    main()
