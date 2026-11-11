#!/usr/bin/env python3
"""Set up a dimer saddle point search from a NEB run. The resulting ``POSCAR``, ``MODECAR`` (and ``INCAR``/``KPOINTS``/``POTCAR``)
are written to ``dim`` by default, or to the directory given with ``-o``.

Usage:
    # cd to the results of NEB
    python3 neb2dim.py -o dim_dir
"""

import argparse
import bz2
import gzip
import os
import re
import shutil
import sys
from subprocess import check_call

import numpy as np
from ase.io import read, write


IMAGE_RE = re.compile(r"^\d\d$")

# Brute force minimum image shifts (reproduces the Vasp.pm pbc routines).
_SHIFTS = np.array(
    [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
    dtype=float,
)


def open_text(path):
    """Open a (possibly gzip/bzip2 compressed) text file."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt")
    return open(path)


def image_dirs(root="."):
    """Return the sorted list of image directories (``00``, ``01``, ...)."""
    dirs = [
        d
        for d in os.listdir(root)
        if IMAGE_RE.match(d) and os.path.isdir(os.path.join(root, d))
    ]
    dirs.sort()
    if not dirs:
        raise SystemExit("No image directories (00, 01, ...) found here.")
    return dirs


def structure_file(image_dir):
    """Prefer a non-empty CONTCAR, otherwise fall back to POSCAR."""
    for name in ("CONTCAR", "POSCAR"):
        path = os.path.join(image_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    raise SystemExit("No CONTCAR/POSCAR in %s" % image_dir)


def outcar_file(image_dir):
    for name in ("OUTCAR", "OUTCAR.gz", "OUTCAR.bz2"):
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return path
    raise SystemExit("No OUTCAR in %s" % image_dir)


def read_structure(image_dir):
    """Read the (relaxed) structure of an image with ASE."""
    return read(structure_file(image_dir), format="vasp")


def outcar_energy_force(image_dir):
    """Return the last ionic step (energy, forces) of an image.

    The forces are taken from the final ``TOTAL-FORCE`` block.  ASE reads
    the selective dynamics flags from CONTCAR/POSCAR, so atoms that are
    fixed have their force components set to zero, exactly as the VTST
    scripts do.
    """
    atoms = read(outcar_file(image_dir), index=-1)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces


def neb_projection(image_dir):
    """Return the VTST NEB projection of the force onto the tangent.

    This is the last number on the final
    ``NEB: projections on to tangent (spring, REAL)`` line.  Endpoint
    images do not have this line, ``None`` is returned for them.
    """
    path = outcar_file(image_dir)
    proj = None
    with open_text(path) as fh:
        for line in fh:
            if "NEB: projections on to tangent" in line:
                proj = float(line.split()[-1])
    return proj


def mic_cartesian(diff, cell):
    """Minimum image cartesian displacement for fractional differences.

    ``diff`` is an (N, 3) array of fractional coordinate differences and
    ``cell`` the (3, 3) cell with lattice vectors as rows.  The search
    over the 27 neighbouring cells reproduces ``pbc_difference_ws`` from
    ``Vasp.pm`` (important for triclinic cells).
    """
    cell = np.asarray(cell, dtype=float)
    candidates = diff[:, None, :] + _SHIFTS[None, :, :]
    cart = candidates @ cell
    dist2 = np.einsum("nkj,nkj->nk", cart, cart)
    best = dist2.argmin(axis=1)
    return cart[np.arange(len(diff)), best]


def path_distances(structures):
    """Distance between consecutive images (the ``dist.pl`` quantity)."""
    distances = [0.0]
    for i in range(1, len(structures)):
        s_prev = structures[i - 1].get_scaled_positions()
        s_cur = structures[i].get_scaled_positions()
        diff = s_cur - s_prev
        cart = mic_cartesian(diff, structures[i].cell)
        distances.append(float(np.sqrt(np.einsum("ij,ij->", cart, cart))))
    return distances


def endpoint_projection(structure, neighbour, forces):
    """Project the endpoint forces onto the local path tangent."""
    cell = np.asarray(structure.cell, dtype=float)
    diff = structure.get_scaled_positions() - neighbour.get_scaled_positions()
    tangent = mic_cartesian(diff, cell)
    norm = np.sqrt(np.einsum("ij,ij->", tangent, tangent))
    if norm == 0.0:
        return 0.0
    tangent /= norm
    return float(np.einsum("ij,ij->", tangent, forces))


def spline_extrema(positions, energies, forces):
    """Find the extrema of the cubic Hermite spline through the MEP.

    Mirrors ``nebspline.pl``: the spline is built from the values
    (``energies``) and the tangential derivatives (``forces``) at each
    image and its derivative is solved for roots inside each interval.

    Returns a list of ``(image_position, energy)`` tuples sorted by
    position, where ``image_position`` is a fractional image index.
    """
    extrema = []
    for i in range(len(positions) - 1):
        dr = positions[i + 1] - positions[i]
        f1 = forces[i] * dr
        f2 = forces[i + 1] * dr
        u1 = energies[i]
        u2 = energies[i + 1]
        fs = f1 + f2
        ud = u2 - u1
        a = u1
        b = -f1
        c = 3.0 * ud + f1 + fs
        d = -2.0 * ud - fs

        disc = c * c - 3.0 * b * d
        if disc < 0.0:
            continue

        roots = []
        if d == 0.0 and c != 0.0:
            roots.append(-(b / (2.0 * c)))
        elif d != 0.0:
            roots.append(-(c + np.sqrt(disc)) / (3.0 * d))
            roots.append(-(c - np.sqrt(disc)) / (3.0 * d))

        for f in roots:
            if 0.0 <= f <= 1.0:
                pos = i + f
                energy = d * f**3 + c * f**2 + b * f + a
                extrema.append((pos, energy))

    extrema.sort(key=lambda item: item[0])
    return extrema


def select_image(extrema, energies, image_override=None):
    """Pick the image (and fraction) at which to start the dimer run."""
    if image_override is not None:
        return int(image_override), float(image_override - int(image_override))

    if extrema:
        position, _ = max(extrema, key=lambda item: item[1])
        index = int(position)
        return index, position - index

    # Monotonic path: fall back to the highest energy image.
    index = int(np.argmax(energies))
    return index, 0.0


def pbc(value):
    """Single-cell periodic wrap used by ``Vasp.pm``."""
    if value <= -0.5:
        value += 1.0
    elif value > 0.5:
        value -= 1.0
    return value


def interpolate_structure(struct_a, struct_b, fraction):
    """Interpolate two images (``posinterp.pl``) and return an ``Atoms``."""
    coords_a = struct_a.get_scaled_positions()
    coords_b = struct_b.get_scaled_positions()

    new = coords_a + fraction * np.vectorize(pbc)(coords_b - coords_a)
    new = np.vectorize(pbc)(new)
    # VASP writes coordinates in the first cell.
    new = np.where(new < 0.0, new + 1.0, new)
    new = np.where(new > 1.0, new - 1.0, new)

    atoms = struct_a.copy()
    atoms.set_scaled_positions(new)
    if "momenta" in atoms.arrays:
        del atoms.arrays["momenta"]
    return atoms


def mode_between(struct_a, struct_b):
    """Difference between two structures as a normalised mode (modemake.pl)."""
    cell = np.asarray(struct_a.cell, dtype=float)
    diff = struct_a.get_scaled_positions() - struct_b.get_scaled_positions()
    diff = np.vectorize(pbc)(diff)
    cart = diff @ cell
    norm = np.sqrt(np.einsum("ij,ij->", cart, cart))
    if norm == 0.0:
        raise SystemExit("Zero distance while building the MODECAR.")
    return cart / norm


def build_mode(structures, index, fraction):
    """Build the MODECAR from the two image pairs around the extremum."""
    n_images = len(structures)

    image_prev = index - 1 if index - 1 >= 0 else index
    image_next = index + 1
    image_next2 = index + 2 if index + 2 < n_images else image_next

    mode_outer = mode_between(structures[image_prev], structures[image_next])
    mode_inner = mode_between(structures[index], structures[image_next2])

    mode = mode_outer * (1.0 - fraction) + mode_inner * fraction
    norm = np.sqrt(np.einsum("ij,ij->", mode, mode))
    return mode / norm


def write_mode(path, mode):
    with open(path, "w") as fh:
        for row in mode:
            fh.write("%20.10E %20.10E %20.10E \n" % (row[0], row[1], row[2]))


def copy_support_files(root, outdir):
    for name in ("INCAR", "KPOINTS"):
        src = os.path.join(root, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(outdir, name))
    for src in (os.path.join(root, "POTCAR"), os.path.join(root, os.pardir, "POTCAR")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(outdir, "POTCAR"))
            break


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert NEB results into dimer input using ASE."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dim",
        help="output directory for the dimer input (default: dim)",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=float,
        default=None,
        help="image (fractional) index to use instead of the highest "
        "energy extremum",
    )
    args = parser.parse_args(argv)

    root = "."
    dirs = image_dirs(root)

    structures = [read_structure(d) for d in dirs]
    energies = []
    forces = []
    projections = []
    for d in dirs:
        energy, force = outcar_energy_force(d)
        energies.append(energy)
        forces.append(force)
        projections.append(neb_projection(d))

    reference = energies[0]
    energies = [e - reference for e in energies]

    # Tangential forces along the MEP (endpoints need explicit projection).
    projections[0] = endpoint_projection(structures[0], structures[1], forces[0])
    projections[-1] = endpoint_projection(
        structures[-1], structures[-2], forces[-1]
    )

    distances = path_distances(structures)
    positions = np.cumsum(distances)

    extrema = spline_extrema(positions, energies, projections)
    index, fraction = select_image(extrema, energies, args.image)

    # The guess is interpolated between images ``index`` and ``index + 1``.
    index = max(0, min(index, len(structures) - 2))

    #print("\n FORMING DIMER BETWEEN IMAGES %d and %d\n" % (index, index + 1))

    guess = interpolate_structure(structures[index], structures[index + 1], fraction)
    mode = build_mode(structures, index, fraction)

    os.makedirs(args.output, exist_ok=True)
    write(os.path.join(args.output, "POSCAR"), guess, format="vasp", direct=True)
    write_mode(os.path.join(args.output, "MODECAR"), mode)
    copy_support_files(root, args.output)

    command = f"sed -i -E '/NSW|IBRION|POTIM|IOPT|MAXMOVE|ICHAIN|SPRING|LCLIMB|IMAGES/d' {os.path.join(args.output, "INCAR")}"
    check_call(command, shell=True)
    command = f"echo 'NSW = 300\nICHAIN = 2\nIOPT = 2\nIBRION = 3\nPOTIM  = 0.0\nEDIFF  = 1.0e-7' >> {os.path.join(args.output, "INCAR")}"
    check_call(command, shell=True)
    print(f"Input files for dimer calculation are generated in {args.output}.")

    # print(" FOR DIMER, REMEMBER TO SET IN THE INCAR: \n")
    # print("ICHAIN = 2 ")
    # print("IOPT = 2 ")
    # print("IBRION = 3 ")
    # print("POTIM  = 0.0 ")
    # print("EDIFF  = 1E-7 \n ")
    # print("DdR       (DEFAULT = 5E-3) ")
    # print("DRotMax   (DEFAULT = 4) ")
    # print("DFNMax    (DEFAULT = 1.0) ")
    # print("DFNMin    (DEFAULT = 0.01) \n")


if __name__ == "__main__":
    sys.exit(main())
