"""
Perturb atomic positions and cell parameters of a structure, input structure should be POSCAR, CONTCAR or extxyz format.

Usage:
# Generate two rattled configurations
python3 rattle_configs.py -i POSCAR -o POSCAR_rattle.extxyz -n 2 

# Generate 2*n rattled configurations, n is the number of structures in the input file, use `-c` option to perturb the cell.
python3 rattle_configs.py -i TiO2.extxyz -o TiO2_rattle.extxyz -n 2 -c
"""

from ase.cell import Cell
from ase.io import read, write 
from ase.neighborlist import NeighborList, natural_cutoffs

import argparse
import numpy as np



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


args = parser.parse_args()

if "POSCAR" in args.infile or "CONTCAR" in args.infile:
    input_structs = [read(args.infile, format="vasp")]
elif args.infile.endwith(".extxyz"):
    input_structs = read(args.infile, index=":")
else:
    print("Error: Only read POSCAR, CONTCAR and extxyz file.")


max_try = 100
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
        out_structs.append(atoms_new)
write(args.outfile, out_structs, format="extxyz")
print(f"{len(out_structs)} structures are written to {args.outfile}.")
