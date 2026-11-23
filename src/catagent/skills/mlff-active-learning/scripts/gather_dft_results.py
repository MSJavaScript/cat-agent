"""
Given a source dir, gather the DFT energies and forces from OUTCARs.

Only ionic steps whose electronic loop converged (|dE| < EDIFF) are kept, and
near-duplicate configurations are removed before writing.

Usage:
# Collect the first 5 converged steps of every OUTCAR under the current dir
python3 gather_dft_results.py --root_dir "./" -o dataset1.extxyz --first 5

# Collect the last 10 steps of every OUTCAR, deduplicate at 0.05 A RMSD
python3 gather_dft_results.py --root_dir runs -o dataset1.extxyz --last 10 --min_dist 0.05
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

LOG = logging.getLogger("gather_dft_results")

DEFAULT_FIRST = 5
COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def _to_float(text: str) -> float:
    """Convert a Fortran-style float (1.0D-03) to a Python float."""
    return float(text.replace("D", "E").replace("d", "e"))


def scan_scf_convergence(outcar: Path) -> tuple[float | None, list[float | None]]:
    """Parse EDIFF and the last SCF energy change of every ionic step.

    ``dE_list[k]`` is the ``total energy-change`` printed just before the k-th
    ``aborting loop`` line, i.e. the convergence measure of the *electronic*
    loop of ionic step ``k`` (not the energy difference between ionic steps).
    ``None`` means that step did not print any energy change (e.g. the SCF loop
    was aborted before the first iteration).

    Returns ``(ediff, dE_list)``. ``ediff`` is ``None`` when the OUTCAR is
    truncated or does not contain the ``stopping-criterion for ELM`` line.
    """
    ediff: float | None = None
    dE_list: list[float | None] = []
    dE: float | None = None

    with open(outcar, "r", errors="ignore") as fh:
        for line in fh:
            if "stopping-criterion for ELM" in line:
                match = _FLOAT_RE.search(line)
                if match:
                    ediff = _to_float(match.group(0))
            elif "total energy-change" in line:
                # " total energy-change (2. order) : 0.123E+02  (-0.456E+01)"
                if ":" in line:
                    match = _FLOAT_RE.search(line.split(":")[-1])
                    if match:
                        dE = _to_float(match.group(0))
            elif "aborting loop" in line:
                dE_list.append(dE)
                dE = None

    return ediff, dE_list


def is_trainable(atoms: Atoms) -> bool:
    """True when the frame carries finite energy and forces."""
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception:
        return False
    return bool(np.isfinite(energy) and np.all(np.isfinite(forces)))


def _select_steps(
    n: int,
    first: int | None,
    last: int | None,
    n_random: int | None,
    rng: random.Random,
    warned: dict[str, bool],
) -> list[int]:
    """Indices of the converged steps to keep from one OUTCAR."""
    if first is None and last is None and n_random is None:
        first = DEFAULT_FIRST

    selected: set[int] = set()
    if first:
        selected.update(range(min(first, n)))
    if last:
        selected.update(range(max(0, n - last), n))
    if n_random:
        if n_random > n and not warned.get("random"):
            LOG.warning(
                "--random %d exceeds the %d converged steps available; taking all of them.",
                n_random,
                n,
            )
            warned["random"] = True
        selected.update(rng.sample(range(n), min(n_random, n)))

    return sorted(selected)


def _displacements(
    frac_new: np.ndarray,
    frac_kept: np.ndarray,
    cells_kept: np.ndarray,
) -> np.ndarray:
    """Minimum-image displacements (cartesian) between one new and K kept structures."""
    delta = frac_new[None, :, :] - frac_kept
    delta -= np.round(delta)
    return np.einsum("kna,kab->knb", delta, cells_kept)


def _distance(cart: np.ndarray, metric: str) -> np.ndarray:
    """Distance of one structure to K kept ones, given their cartesian displacements."""
    squared = np.einsum("kna,kna->kn", cart, cart)
    if metric == "max":
        return np.sqrt(squared.max(axis=1))
    total = squared.sum(axis=1)
    if metric == "total":
        return np.sqrt(total)
    return np.sqrt(total / cart.shape[1])  # rmsd


def _grow(array: np.ndarray, capacity: int) -> np.ndarray:
    grown = np.empty((capacity,) + array.shape[1:], dtype=array.dtype)
    grown[: len(array)] = array
    return grown


class _Group:
    """Kept structures with an identical composition, stored for vectorized compares."""

    __slots__ = ("size", "frac", "cells")

    def __init__(self, natoms: int, capacity: int = 64):
        self.size = 0
        self.frac = np.empty((capacity, natoms, 3))
        self.cells = np.empty((capacity, 3, 3))

    def append(self, frac: np.ndarray, cell: np.ndarray) -> None:
        if self.size == len(self.frac):
            capacity = 2 * len(self.frac)
            self.frac = _grow(self.frac, capacity)
            self.cells = _grow(self.cells, capacity)
        self.frac[self.size] = frac
        self.cells[self.size] = cell
        self.size += 1


def deduplicate(configs: list[Atoms], min_dist: float, metric: str) -> list[int]:
    """Greedily drop configurations closer than ``min_dist`` to a kept one.

    Structures are compared only when they have the same number of atoms, the
    same species (in the same order) and the same cell; a new structure is kept
    when it is farther than ``min_dist`` from *all* already kept structures.
    This avoids the chaining problem of single-linkage clustering, where a
    chain of pairwise-close structures collapses into a single one.

    The metric is ``rmsd`` (per-atom root-mean-square displacement, A),
    ``max`` (largest atomic displacement, A) or ``total`` (sqrt of the summed
    squared displacements, A).
    """
    groups: dict[tuple[int, tuple[str, ...]], _Group] = {}
    kept: list[int] = []

    for index, atoms in enumerate(configs):
        symbols = tuple(atoms.get_chemical_symbols())
        key = (len(atoms), symbols)
        group = groups.get(key)
        if group is None:
            group = groups[key] = _Group(len(atoms))

        frac = atoms.get_scaled_positions()
        cell = np.asarray(atoms.get_cell())

        if group.size:
            kept_frac = group.frac[: group.size]
            kept_cells = group.cells[: group.size]
            cart = _displacements(frac, kept_frac, kept_cells)
            distance = _distance(cart, metric)
            same_cell = np.all(np.isclose(kept_cells, cell), axis=(1, 2))
            if np.any(same_cell & (distance < min_dist)):
                continue

        group.append(frac, cell)
        kept.append(index)

    return kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gather the DFT energies and forces from VASP OUTCARs."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
        help="Root dir of VASP results. Default is the current directory.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help="The name of extxyz file to save the configurations.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="OUTCAR*",
        help="Glob pattern (relative to --root_dir) used to find the OUTCARs. Default OUTCAR*.",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="Two structures closer than this value (in the unit of --metric) are "
        "considered identical. Default 0.1.",
    )
    parser.add_argument(
        "--metric",
        choices=("rmsd", "max", "total"),
        default="rmsd",
        help="Distance used for deduplication: 'rmsd' (per-atom RMSD), 'max' "
        "(largest atomic displacement) or 'total' (sqrt of the summed squared "
        "displacements, no size normalization). Default rmsd.",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=None,
        help=f"Select the first n converged steps from each OUTCAR. Default {DEFAULT_FIRST} "
        "when neither --first, --last nor --random is given.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Select the last n converged steps from each OUTCAR.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Randomly select n converged steps from each OUTCAR.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --random. Default 42, so runs are reproducible.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-file progress.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    for name in ("first", "last", "random"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name} must be >= 0")
    if args.min_dist < 0:
        parser.error("--min_dist must be >= 0")

    root = Path(args.root_dir)
    if not root.is_dir():
        LOG.error("--root_dir %s is not a directory.", root)
        return 1

    outcars = sorted(
        path
        for path in root.rglob(args.pattern)
        if path.is_file() and path.suffix not in COMPRESSED_SUFFIXES
    )
    if not outcars:
        LOG.error("No files matching %r found under %s.", args.pattern, root)
        return 1

    rng = random.Random(args.seed)
    warned: dict[str, bool] = {}

    collected: list[Atoms] = []
    n_frames = 0
    n_converged = 0
    n_selected = 0
    n_skipped_files = 0
    n_unconverged = 0

    for outcar in outcars:
        try:
            frames = read(outcar, index=":", format="vasp-out")
        except Exception as exc:
            LOG.warning("Skipping %s: cannot read OUTCAR (%s).", outcar, exc)
            n_skipped_files += 1
            continue

        if isinstance(frames, Atoms):
            frames = [frames]
        if not frames:
            LOG.warning("Skipping %s: no ionic steps found.", outcar)
            n_skipped_files += 1
            continue

        for atoms in frames:
            atoms.set_constraint([])

        try:
            ediff, dE_list = scan_scf_convergence(outcar)
        except OSError as exc:
            LOG.warning("Skipping %s: cannot parse OUTCAR (%s).", outcar, exc)
            n_skipped_files += 1
            continue

        if ediff is None:
            LOG.warning(
                "Skipping %s: no 'stopping-criterion for ELM' line, cannot judge "
                "electronic convergence (truncated OUTCAR?).",
                outcar,
            )
            n_skipped_files += 1
            continue

        if ediff <= 0:
            LOG.warning("Skipping %s: non-positive EDIFF (%g).", outcar, ediff)
            n_skipped_files += 1
            continue

        n_pairs = min(len(dE_list), len(frames))
        if len(dE_list) != len(frames):
            LOG.warning(
                "%s: %d ionic frames but %d SCF summaries; only the first %d frames "
                "can be checked and the remaining %d are dropped.",
                outcar,
                len(frames),
                len(dE_list),
                n_pairs,
                len(frames) - n_pairs,
            )

        good: list[int] = []
        for i in range(n_pairs):
            if dE_list[i] is None or abs(dE_list[i]) >= ediff:
                continue
            if not is_trainable(frames[i]):
                continue
            good.append(i)

        n_frames += len(frames)
        n_converged += len(good)
        n_unconverged += n_pairs - len(good)

        if not good:
            LOG.warning("%s: no converged ionic step with energy and forces.", outcar)
            continue

        selected = _select_steps(
            len(good), args.first, args.last, args.random, rng, warned
        )
        for position in selected:
            index = good[position]
            atoms = frames[index]
            atoms.info["config_type"] = "dft"
            atoms.info["source"] = str(outcar)
            atoms.info["ionic_step"] = int(index)
            collected.append(atoms)
            n_selected += 1

        if args.verbose:
            LOG.info(
                "%s: %d/%d ionic steps converged, %d selected.",
                outcar,
                len(good),
                len(frames),
                len(selected),
            )

    if not collected:
        LOG.error(
            "No structures collected from %d OUTCAR file(s). Nothing written.",
            len(outcars),
        )
        return 1

    kept = deduplicate(collected, args.min_dist, args.metric)
    result = [collected[i] for i in kept]

    try:
        write(args.outfile, result, format="extxyz")
    except Exception as exc:
        LOG.error("Cannot write %s: %s", args.outfile, exc)
        return 1

    metric_label = {
        "rmsd": "per-atom RMSD",
        "max": "maximum atomic displacement",
        "total": "total displacement",
    }[args.metric]

    print(f"Scanned {len(outcars)} OUTCAR(s), skipped {n_skipped_files}.")
    print(
        f"Converged ionic steps: {n_converged}/{n_frames} "
        f"({n_unconverged} unconverged or incomplete dropped)."
    )
    print(
        f"Selected {n_selected}; removed {n_selected - len(result)} duplicate(s) "
        f"closer than {args.min_dist:g} A ({metric_label})."
    )
    print(f"{len(result)} structures are gathered and written to {args.outfile}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
