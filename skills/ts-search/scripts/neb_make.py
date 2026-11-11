"""
Interpolate between initial state and final state, generate the initial NEB path.

Usage:
python3 neb_make.py --ini POSCAR_ini --fin POSCAR_fin --spacing 0.5 
python3 neb_make.py --ini POSCAR_ini --fin POSCAR_fin --nimages 5 --exyxyz neb_path.extxyz
"""

from ase.io import read, write 
from ase.mep import NEB 
import numpy as np
import argparse, os


def get_structures_distance(atoms1, atoms2)->float:
    if len(atoms1)!= len(atoms2):
        return -1.0
    cell1 = atoms1.get_cell()
    cell2 = atoms2.get_cell()
    if not np.all(np.isclose(cell1, cell2)):
        return -1.0
    frac1 = atoms1.get_scaled_positions()
    frac2 = atoms2.get_scaled_positions()
    diff_frac = frac1 - frac2
    diff_frac[diff_frac > 0.5] -= 1.0
    diff_frac[diff_frac < -0.5] += 1.0
    diff_cart = np.matmul(diff_frac, cell1)
    return np.sum(diff_cart**2.0) ** 0.5 

parser = argparse.ArgumentParser(description='Make an initial guess of NEB path from initial and final state.')
parser.add_argument("--ini", type=str, required=True, help="Path to initial state POSCAR or CONTCAR.")
parser.add_argument("--fin", type=str, required=True, help="Path to final state POSCAR or CONTCAR.")
parser.add_argument("--spacing", type=float, default=0.5, help="The spacing between images in the NEB path, this will be ignored if `--nimages` is set. Default 0.5.")
parser.add_argument("--nimages", type=int, help="Number of intermediate images.")
parser.add_argument("--extxyz", type=str, help="Name of extxyz file to save the NEB path, if ignored, images will be written into 0x dirs as VASP input.")

args = parser.parse_args()

ini = args.ini 
fin = args.fin 
spacing = args.spacing
nimages = args.nimages
extxyz = args.extxyz

poscar_ini = read(ini)
poscar_fin = read(fin)

if nimages is None:
    d = get_structures_distance(poscar_ini, poscar_fin)
    nimages = int(d / spacing) - 1
    
images = [poscar_ini]
images += [poscar_ini.copy() for _ in range(nimages)]
images.append(poscar_fin)
neb = NEB(images, method='improvedtangent')
neb.interpolate(method='idpp', mic=True)

if extxyz:
    write(extxyz, neb.images, format="extxyz")
    print(f"NEB path is saved in {extxyz}.")
else:
    for i in range(len(neb.images)):
        dirname = f'0{i}' if i < 10 else str(i)
        os.makedirs(dirname, exist_ok=True)
        write(os.path.join(dirname, "POSCAR"), neb.images[i], format="vasp")
    
    print(f"NEB images are generated in dir 00 to 0{i}.")
 