"""
Fix the atomic positions of slab atoms for frequency analysis.

Usage:
    python3 fix_slab_atoms.py --in POSCAR --out POSCAR_fix --relax_ind "30,29,28"
"""

import argparse
from ase.io import read, write 
from ase.constraints import FixAtoms

parser = argparse.ArgumentParser(description="Fix atomic positions of slab atoms.")
parser.add_argument("-i", "--infile", type=str, required=True, help="Path to input POSCAR or CONTCAR.")
parser.add_argument("-o", "--outfile", type=str, required=True, help="Name of output POSCAR.")
parser.add_argument("--relax_ind", type=str, help="Indices of atoms (start from 0) which are allowed to relaxed, separated by comma. If omitted, fix all the atoms except C,H,O,N. ")
args = parser.parse_args()

poscar_in = read(args.infile)
natoms = len(poscar_in)
symbols = poscar_in.get_chemical_symbols()

if args.relax_ind is None:
    inds = [i for i in range(natoms) if not symbols[i] in ['C', 'H', 'O', 'N']]
else:
    relax_inds = args.relax_ind.split(",")
    relax_inds = [int(a.strip()) for a in relax_inds]
    inds = list(set(range(natoms)).difference(relax_inds))

poscar_in.set_constraint(FixAtoms(indices=inds))
write(args.outfile, poscar_in, format="vasp")
print(f"{args.outfile} is generated.")

