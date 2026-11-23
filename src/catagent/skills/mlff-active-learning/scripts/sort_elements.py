"""
Sort the elements of POSCAR. When you call ase `.repeat()` method to enlarge the cell or append atoms to a structure, the order of elements are not sorted. For example, if you repeat SiO2 4 times, the order of elements may become "Si O Si O Si O Si O". This script sorts the elements. 

Usage:
python3 sort_elements.py -i POSCAR -o POSCAR_sorted
"""

import numpy as np
from ase.io import read, write
import sys
from ase.data import atomic_numbers
from ase import Atoms
from ase.constraints import FixAtoms
import argparse

parser = argparse.ArgumentParser(description="Sort the elements in a POSCAR.")
parser.add_argument("-i", "--infile", type=str, required=True, help="Input structure, should be POSCAR or CONTCAR.")
parser.add_argument("-o", "--outfile", type=str, required=True, help="Name of output POSCAR.")
args = parser.parse_args()

poscar = read(args.infile, format="vasp")

pos = poscar.get_positions()
natoms = pos.shape[0]
atoms_count = poscar.symbols.formula.count()
index_count = {key:[] for key in atoms_count.keys()}
symbols = poscar.get_chemical_symbols()
for i, s in enumerate(symbols):
    index_count[s].append(i)

numbers = []
index = []
for key in index_count.keys():
    numbers.extend([atomic_numbers[key]] * len(index_count[key]))
    index.extend(index_count[key])

constraints = None
if len(poscar.constraints) != 0:
    fix_indices = poscar.constraints[0].get_indices()
    flag = np.zeros((natoms, ), dtype=bool)
    flag[fix_indices] = True
    new_flag = flag[index]
    new_fix_indices = np.where(new_flag)[0].tolist()
    constraints = FixAtoms(indices=new_fix_indices)

new_pos = pos[index, :]
new_atoms = Atoms(numbers = numbers,
                  positions = new_pos,
                  cell = poscar.get_cell())

if not constraints is None:
    new_atoms.set_constraint(constraints)

write(args.outfile, new_atoms, format="vasp")
print(f"The elements are sorted, structure is written into {args.outfile}.")





