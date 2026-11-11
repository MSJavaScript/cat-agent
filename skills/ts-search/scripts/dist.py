"""
Calculate the distance between two structures: 
Usage: python3 dist.py POSCAR_1 POSCAR_2
"""

from ase.io import read 
import numpy as np
import sys


def get_structures_distance(atoms1, atoms2)->float:
    if len(atoms1)!= len(atoms2):
        raise ValueError("The number of atoms in two structures are not the same.")
    cell1 = atoms1.get_cell()
    cell2 = atoms2.get_cell()
    if not np.all(np.isclose(cell1, cell2)):
        raise ValueError("The cells of two structures are not the same.")
    
    frac1 = atoms1.get_scaled_positions()
    frac2 = atoms2.get_scaled_positions()
    diff_frac = frac1 - frac2
    diff_frac[diff_frac > 0.5] -= 1.0
    diff_frac[diff_frac < -0.5] += 1.0
    diff_cart = np.matmul(diff_frac, cell1)
    return np.sum(diff_cart**2.0) ** 0.5 


if len(sys.argv) != 3:
    print("Calculate the distance between two structures: ")
    print("Usage: python3 dist.py POSCAR_1 POSCAR_2")
    exit(1)

atoms1 = read(sys.argv[1])
atoms2 = read(sys.argv[2])
d = get_structures_distance(atoms1, atoms2)
print(f"Distance (Angstrom): {d:.2f}")

