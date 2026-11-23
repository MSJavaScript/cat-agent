"""
Extract structures from structural relaxation, NEB or MD trajectories.

If the number of ionic steps in the trajectory is larger than the given
threshold, the trajectory is considered problematic and structures are sampled
from it for DFT labelling.

Usage:
# Extract structures from a structural relaxation
python3 check_traj.py --traj opt.traj -o extracted_opt.extxyz -n 5

# Extract structures from an NEB trajectory. Two end points are ignored,
# because they are IS and FS, which keep fixed during the relaxation.
python3 check_traj.py --traj opt.traj -o extracted_neb.extxyz --neb -n 5

Exit code is 0 when no structure needs to be extracted (the trajectory is
considered converged) and 1 when the input is broken or cannot be read.

The written extxyz contains geometry only: energies and forces computed by the
MLFF are not copied, so the file can be labelled by DFT without stale labels.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

import numpy as np
from ase import Atoms
from ase.io import read, write

LOG = logging.getLogger("check_traj")

PERIOD_TOL = 1.0e-6  # Angstrom, used to detect repeated frames of an NEB


def get_structures_distance(atoms1: Atoms, atoms2: Atoms) -> float:
    """Total displacement (Angstrom) between two comparable structures.

    Structures are comparable when they have the same number of atoms, the same
    species in the same order, and the same cell. Otherwise ``np.nan`` is
    returned and the pair is never treated as identical.

    The fractional differences are wrapped into [-0.5, 0.5). For the small
    displacements compared here this is a good approximation of the minimum
    image convention (it is exact for orthogonal and near-orthogonal cells).
    """
    if len(atoms1) != len(atoms2):
        return np.nan
    if atoms1.get_chemical_symbols() != atoms2.get_chemical_symbols():
        return np.nan

    cell1 = atoms1.get_cell()
    cell2 = atoms2.get_cell()
    if not np.all(np.isclose(cell1, cell2)):
        return np.nan

    diff = atoms1.get_scaled_positions() - atoms2.get_scaled_positions()
    diff -= np.round(diff)
    return float(np.sqrt(np.sum(np.matmul(diff, cell1) ** 2)))


def detect_nimages(
    frames: list[Atoms], tol: float = PERIOD_TOL
) -> tuple[int | None, list[int]]:
    """Infer the number of NEB images from a flattened ASE NEB trajectory.

    ASE writes one frame per image per optimizer step, so the initial state
    reappears every ``nimages`` frames. A candidate period is accepted only if
    the initial state also reappears at multiples of it, which rejects most
    coincidental matches of an intermediate image.

    Returns ``(nimages, repeated_indices)``; ``nimages`` is ``None`` when the
    period cannot be determined.
    """
    repeats: list[int] = []
    for i in range(1, len(frames)):
        distance = get_structures_distance(frames[0], frames[i])
        if np.isfinite(distance) and distance < tol:
            repeats.append(i)

    repeat_set = set(repeats)
    for period in repeats:
        if period < 2 or len(frames) % period:
            continue
        nblocks = len(frames) // period
        if nblocks < 2:
            continue
        # The initial state must recur at every multiple of the period.
        if all(k * period in repeat_set for k in range(1, min(nblocks, 4))):
            return period, repeats

    return None, repeats


def choose_indices(
    n: int, k: int, rng: random.Random, method: str = "random"
) -> list[int]:
    """Choose at most ``k`` indices out of ``range(n)``, sorted and unique."""
    k = min(max(k, 0), n)
    if k == 0:
        return []
    if k == n:
        return list(range(n))
    if method == "uniform":
        indices = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
        return [int(i) for i in indices]
    return sorted(rng.sample(range(n), k))


def clean_structure(
    atoms: Atoms, source: str, image: int | None = None, step: int | None = None
) -> None:
    """Drop MLFF results and attach provenance metadata.

    The structure is written for DFT labelling, so the energy and forces of the
    MLFF are removed. The MLFF energy is kept as ``mlff_energy`` for reference.
    """
    if atoms.calc is not None:
        energy = atoms.calc.results.get("energy")
        if energy is not None and np.isfinite(energy):
            atoms.info["mlff_energy"] = float(energy)
    atoms.calc = None

    atoms.info["source"] = str(source)
    if image is not None:
        atoms.info["image"] = int(image)
    if step is not None:
        atoms.info["step"] = int(step)


def extract_opt(
    frames: list[Atoms], args: argparse.Namespace, rng: random.Random
) -> list[Atoms]:
    nstruct = len(frames)
    if nstruct < args.nstep_thres:
        return []

    k = min(args.n_extract, nstruct)
    if k < args.n_extract:
        LOG.warning(
            "Only %d steps available; extracting %d instead of %d.",
            nstruct,
            k,
            args.n_extract,
        )

    result: list[Atoms] = []
    for step in choose_indices(nstruct, k, rng, args.sample):
        atoms = frames[step]
        clean_structure(atoms, args.traj, step=step)
        result.append(atoms)
    return result


def extract_neb(
    frames: list[Atoms], args: argparse.Namespace, rng: random.Random
) -> list[Atoms] | None:
    """Extract structures from an NEB trajectory; ``None`` means a broken input."""
    nstruct = len(frames)
    nimages = args.nimages

    if nimages is None:
        nimages, repeats = detect_nimages(frames)
        if nimages is None:
            hint = ", ".join(str(i) for i in repeats[:8])
            LOG.error(
                "Cannot infer the number of NEB images from %s. Frames equal to the "
                "initial state were found at indices: [%s]. If the path has images "
                "that coincide with the initial state, pass --nimages explicitly.",
                args.traj,
                hint,
            )
            return None
        if args.verbose:
            LOG.info("Detected %d NEB images.", nimages)

    if nimages < 2:
        LOG.error("--nimages must be >= 2 for an NEB trajectory (got %d).", nimages)
        return None
    if nstruct % nimages:
        LOG.error(
            "The trajectory has %d frames, which is not divisible by the %d NEB "
            "images; it is probably truncated.",
            nstruct,
            nimages,
        )
        return None

    nsteps = nstruct // nimages
    if nsteps < args.nstep_thres:
        return []
    if nimages < 3:
        LOG.warning(
            "The NEB path has only %d images (IS and FS); no intermediate image "
            "can be extracted.",
            nimages,
        )
        return []

    k = min(args.n_extract, nsteps)
    if k < args.n_extract:
        LOG.warning(
            "Only %d step(s) per image; extracting %d instead of %d.",
            nsteps,
            k,
            args.n_extract,
        )

    result: list[Atoms] = []
    for image in range(1, nimages - 1):
        sub_traj = frames[image:nstruct:nimages]
        for step in choose_indices(len(sub_traj), k, rng, args.sample):
            atoms = sub_traj[step]
            clean_structure(atoms, args.traj, image=image, step=step)
            result.append(atoms)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the trajectory and extract structures if the trajectory "
        "has too many ionic steps."
    )
    parser.add_argument(
        "--traj",
        type=str,
        required=True,
        help="Path to a *.traj file.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help="The name of extxyz file to save the extracted structures.",
    )
    parser.add_argument(
        "--neb",
        action="store_true",
        help="If this *.traj file is generated by an NEB job.",
    )
    parser.add_argument(
        "--nimages",
        type=int,
        default=None,
        help="Number of NEB images, including IS and FS. Only used with --neb. "
        "Inferred from repeated frames when omitted.",
    )
    parser.add_argument(
        "--nstep_thres",
        type=int,
        default=100,
        help="If the number of steps in the trajectory is not smaller than this "
        "value, structures are extracted from it. Default 100.",
    )
    parser.add_argument(
        "-n",
        "--n_extract",
        type=int,
        default=3,
        help="Number of structures to extract. For NEB this is the number per "
        "intermediate image; the two end points (IS and FS) are ignored. Default 3.",
    )
    parser.add_argument(
        "-s",
        "--sample",
        choices=("random", "uniform"),
        default="random",
        help="How structures are picked: 'random' (default) or 'uniform' "
        "(evenly spaced over the trajectory, always including the last step).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --sample random. Default 42, so runs are reproducible.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.n_extract < 1:
        parser.error("--n_extract must be >= 1")
    if args.nstep_thres < 0:
        parser.error("--nstep_thres must be >= 0")
    if args.nimages is not None:
        if not args.neb:
            parser.error("--nimages is only valid together with --neb")
        if args.nimages < 2:
            parser.error("--nimages must be >= 2")

    try:
        frames = read(args.traj, index=":")
    except Exception as exc:
        LOG.error("Cannot read trajectory %s: %s", args.traj, exc)
        return 1

    if isinstance(frames, Atoms):
        frames = [frames]
    if not frames:
        LOG.error("Trajectory %s contains no frames.", args.traj)
        return 1

    rng = random.Random(args.seed)
    result = (
        extract_neb(frames, args, rng) if args.neb else extract_opt(frames, args, rng)
    )

    if result is None:
        return 1
    if not result:
        print("No structures are extracted.")
        return 0

    try:
        write(args.outfile, result, format="extxyz")
    except Exception as exc:
        LOG.error("Cannot write %s: %s", args.outfile, exc)
        return 1

    print(f"{len(result)} structures are extracted and saved to {args.outfile}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
