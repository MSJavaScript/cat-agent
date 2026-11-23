"""
Convert a conventional cell to primitive cell or vice versa.

Usage:
# Conver conventional cell to primitive cell
python3 conv_prim.py --conv POSCAR -o POSCAR_prim

# Conver primitive cell to conventional cell
python3 conv_prim.py --prim structure.cif -o conv.cif
"""
import argparse
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import (
    ConventionalCellTransformation, PrimitiveCellTransformation
)


parser = argparse.ArgumentParser(description="Convert a conventional cell to primitive cell or vice versa.")
parser.add_argument("--conv", type=str, help="A source structure containing a conventional cell.")
parser.add_argument("--prim", type=str, help="A source structure containing a primitive cell.")
parser.add_argument("-o", "--outfile", type=str, help="The name of output file, should start with 'POSCAR'.")

symprec = 0.1
angle_tolerance = 5.0 

args = parser.parse_args()
if args.prim:
    structure = Structure.from_file(args.prim)
    conv_trans = ConventionalCellTransformation(symprec=symprec, angle_tolerance=angle_tolerance)
    result = conv_trans.apply_transformation(structure)
    result.to(args.outfile)
    print(f"{args.outfile} is generated.")
elif args.conv:
    structure = Structure.from_file(args.conv)
    prim_trans = PrimitiveCellTransformation()
    result = prim_trans.apply_transformation(structure)
    result.to(args.outfile)
    print(f"{args.outfile} is generated.")
else:
    print("Error: You must set the source file name by --prim or --conv options.")
    exit(0)
