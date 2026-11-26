"""
Perturb atomic positions and cell parameters of a structure, input structure should be POSCAR, CONTCAR or extxyz format.

Usage:
# Generate two rattled configurations
python3 rattle_configs.py -i POSCAR -o POSCAR_rattle.extxyz -n 2 

# Generate 2*n rattled configurations, n is the number of structures in the input file, use `-c` option to perturb the cell.
python3 rattle_configs.py -i TiO2.extxyz -o TiO2_rattle.extxyz -n 2 -c

# Also push apart overlapping atoms: no pair is left closer than
# (covalent radius of atom 1 + covalent radius of atom 2 - 0.3 A).
python3 rattle_configs.py -i POSCAR -o POSCAR_rattle.extxyz -n 2 --min_dist_delta 0.3
"""

from ase.cell import Cell
from ase.io import read, write 
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list
from ase.data import covalent_radii
import argparse
import numpy as np


def adjust_close_atoms(atoms, delta, max_iter=200):
    """Push apart atoms closer than the sum of their covalent radii minus delta.

    A pair (i, j) is too close when

        d_ij < covalent_radii[i] + covalent_radii[j] - delta

    with ``d_ij`` the minimum-image distance.  For every such pair the two
    atoms are moved apart by half of the deficit along their bond vector, and
    the check is repeated because moving one pair can create or remove other
    overlaps.

    ``delta`` therefore controls how much closer than a "normal" bond a pair
    may be: it must be larger than ``r_i + r_j - d_ij`` of a typical bond,
    otherwise ordinary bonds are flagged as overlaps.  (Covalent radii
    overestimate metallic bond lengths, so metals need a larger delta than
    covalent compounds.)

    The adjustment is done in place and does not wrap the positions.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose positions are adjusted.
    delta : float
        Tolerance in Angstrom: only overlaps deeper than ``delta`` are
        corrected, and pairs are separated to ``r_i + r_j - delta``.
    max_iter : int
        Maximum number of correction rounds. Default 200.

    Returns
    -------
    float
        Largest remaining overlap (Angstrom) after ``max_iter`` rounds;
        ``0.0`` when no pair is too close.
    """
    radii = covalent_radii[atoms.numbers]
    tol = 1e-8  # Angstrom; ignore numerical noise at the threshold

    # No pair can be closer than r_i + r_j - delta <= 2 * max(r) - delta,
    # so this cutoff is enough to find every overlap.
    cutoff = 2.0 * radii.max() - delta
    if cutoff <= 0.0:
        return 0.0

    def close_pairs():
        i, j, d, D = neighbor_list("ijdD", atoms, cutoff, self_interaction=False)
        # neighbor_list returns both (i, j) and (j, i); keep each pair once.
        keep = i < j
        i, j, d, D = i[keep], j[keep], d[keep], D[keep]
        limit = radii[i] + radii[j] - delta
        too_close = d < limit - tol
        return i[too_close], j[too_close], d[too_close], D[too_close], limit[too_close]

    i, j, d, D, limit = close_pairs()
    for _ in range(max_iter):
        if len(i) == 0:
            return 0.0

        deficit = limit - d
        unit = np.zeros_like(D)
        nonzero = d > 0.0
        unit[nonzero] = D[nonzero] / d[nonzero, None]
        # Exactly coincident atoms have no bond direction; separate them along x.
        unit[~nonzero] = (1.0, 0.0, 0.0)

        # D points from i to j: move i backwards and j forwards.
        shift = 0.5 * deficit[:, None] * unit
        displacement = np.zeros_like(atoms.positions)
        np.add.at(displacement, i, -shift)
        np.add.at(displacement, j, shift)
        atoms.set_positions(atoms.get_positions() + displacement)

        i, j, d, D, limit = close_pairs()

    if len(i) == 0:
        return 0.0
    return float(np.max(limit - d))



parser = argparse.ArgumentParser(description="Perturb atomic positions and cell parameters of a structure.")
parser.add_argument("-i", "--infile", type=str, required=True, help="Input structure, should be POSCAR, CONTCAR or extxyz file.")
parser.add_argument("-o", "--outfile", type=str, required=True, help="Name of an extxyz file to save the output perturbed structures.")
parser.add_argument("-n", "--n_perturb", type=int, default=1, help="The number of perturbed structures generated from one input structure. Default 1.")

parser.add_argument("-c", "--cell_perturb", 
                    action="store_true", 
                    help="If the cell is perturbed.")

parser.add_argument("--dx", type=float, 
                    default=0.25, 
                    help="The change in atomic position when perturbing atomic coordinates. Default 0.25.")

parser.add_argument("--dv", 
                    type=float, 
                    default=0.1, 
                    help="The percentage of change in unit cell volume when lattice constant is perturbed. In the range of [0, 0.3], default is 0.1.")

parser.add_argument("--min_dist_delta", 
                    type=float, 
                    default=None, 
                    help="If set, atoms closer than the sum of their covalent radii minus this value (Angstrom) are pushed apart. Default None, i.e. no overlap check.")


args = parser.parse_args()

if args.min_dist_delta is not None and args.min_dist_delta < 0:
    parser.error("--min_dist_delta must be >= 0")

if "POSCAR" in args.infile or "CONTCAR" in args.infile:
    input_structs = [read(args.infile, format="vasp")]
elif args.infile.endswith(".extxyz"):
    input_structs = read(args.infile, index=":")
else:
    print("Error: Only read POSCAR, CONTCAR and extxyz file.")


max_try = 100
max_adjust_iter = 200
out_structs = []
for atoms in input_structs:

    n_atoms = len(atoms)
    n_adj_list = []
    nl = NeighborList(cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    for i in range(n_atoms):
        indices, offsets = nl.get_neighbors(i)
        n_adj_list.append(len(indices))

    n_adj = np.array(n_adj_list, dtype=float)
    mass = atoms.get_masses()
    score = (1.0 - n_adj/np.max(n_adj))+ (1.0 - mass/np.max(mass))

    if len(atoms.constraints) > 0:
        fix_index = atoms.constraints[0].get_indices()
        score[fix_index] = 0
    
    for i in range(args.n_perturb):

        cell = atoms.get_cell()
        if args.cell_perturb:
            a,b,c,_,_,_ = cell.cellpar()
            s = ((1 + abs(args.dv)) ** (1.0/3.0) - 1.0)*max(a,b,c)
            for j in range(max_try):
                new_cell = Cell(cell + (np.random.random((3,3))-0.5)*s)
                if abs(new_cell.volume - cell.volume)/cell.volume < abs(args.dv):
                    cell = new_cell
                    break 
            else:
                cell = new_cell
        
        atoms_new = atoms.copy()
        atoms_new.set_constraint([])
        atoms_new.set_cell(cell)

        vec = np.random.random((n_atoms, 3))
        vec[:, 0] += 1.0e-6
        vec = vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]
        positions = atoms_new.get_positions() \
            + vec*abs(args.dx)*score[:, np.newaxis] \
            + (np.random.random((n_atoms, 3)) - 0.5)*abs(args.dx)
        atoms_new.set_positions(positions)

        if args.min_dist_delta is not None:
            remaining = adjust_close_atoms(
                atoms_new, args.min_dist_delta, max_iter=max_adjust_iter
            )
            if remaining > 0.05:
                print(
                    f"Warning: structure {len(out_structs)} still has atom "
                    f"pairs up to {remaining:.2f} A closer than requested "
                    f"after {max_adjust_iter} rounds; --min_dist_delta may be "
                    "too small for this system (normal bonds are treated as "
                    "overlaps), or the cell is too compressed."
                )

        out_structs.append(atoms_new)
write(args.outfile, out_structs, format="extxyz")
print(f"{len(out_structs)} structures are written to {args.outfile}.")
