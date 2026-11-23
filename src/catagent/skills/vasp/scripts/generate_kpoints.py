"""
Generate KPOINTS for normal vasp job or band structure calculation.
Usage:

# Set the number of kpoints along a,b,c directions
python3 generate_kpoints.py --poscar POSCAR -o KPOINTS mesh --abc '5,5,3' -s gamma

# Set the kpoints mesh by kspacing
python3 generate_kpoints.py --poscar POSCAR mesh --kspacing 0.25

# print special kpoints
python3 generate_kpoints.py --poscar POSCAR line --pbc '1,1,1'

# Generate KPOINTS by line mode
python3 generate_kpoints.py --poscar POSCAR line --pbc '1,1,1' --kpath 'GXXWWG'
"""

from pymatgen.io.vasp import Kpoints 
import math, argparse
from ase.io import read 
import numpy as np
from fractions import Fraction

def non_periodic_flags(poscar):
    cellpar = poscar.cell.cellpar()
    direct_pos = poscar.get_scaled_positions(wrap=True)
    vacuum_size_direct = 1 - (np.max(direct_pos, axis=0) - np.min(direct_pos, axis=0))
    vacuum_size = vacuum_size_direct * np.array([cellpar[0], cellpar[1], cellpar[2]])
    return vacuum_size > 5.0

parser = argparse.ArgumentParser(description="Generate KPOINTS file.")
parser.add_argument("--poscar", required=True, type=str, help="Path to POSCAR file.")
parser.add_argument("-o", "--outfile", type=str, default="KPOINTS", help="Name of output file. Default is KPOINTS.")

subparsers = parser.add_subparsers(dest='type', help='KPOINTS type. Regular mesh type or line type (for band structure calculation).')

mesh_parser = subparsers.add_parser('mesh', help='Regular mesh type KPOINTS.')
mesh_parser.add_argument("-s", "--scheme", type=str, default="gamma", help="The type of mesh, Gamma-centered (gamma) or Monkhorst-Pack (mp). Default is gamma.")
mesh_parser.add_argument("--abc", type=str, help="The number of kpoints along a,b,c directions, separated by comma. e.g. '5,5,3'. ")
mesh_parser.add_argument("--shift", type=str, default="0,0,0", help="Shifts of kpoints along a,b,c directions. Default is '0,0,0'.")
mesh_parser.add_argument("--kspacing", type=float, help="Spacing of kpoints in reciprocal space. The number of k points in the direction of reciprocal lattice vector  `b_i`  is determined by `max(1, ceil(|b_i|*2*pi/kspacing))`. This argument is ignored if `--abc` option is set.")


line_parser = subparsers.add_parser('line', help='Line type KPOINTS for band structure.')
line_parser.add_argument("--pbc", type=str, default="1,1,1", help="Periodic condition flags along a,b,c directions. Three 1 or 0 integers separated by comma. Default is '1,1,1'. If only this option is set, high-symmetry kpoints will be printed.")
line_parser.add_argument("-n", "--nkpoints", type=int, help="The number of kpoints along each high symmetry line.")
line_parser.add_argument("--kpath", type=str, help="High-symmetry kpoints paths, for example, 'GXXWWG'. If this option is None, it will be generated automatically. Use 'G' for Gamma point.")

args = parser.parse_args()
outfile  = args.outfile
poscar = read(args.poscar, format="vasp")

if args.type == 'mesh':
    scheme   = args.scheme
    abc      = args.abc
    kspacing = args.kspacing
    shift = args.shift
    shift = [float(s) for s in shift.split(',')]
    if abc is None and kspacing is None:
        print("Error: You must set `--abc` option or `--kspacing` option.")
        exit(0)

    flags = non_periodic_flags(poscar)
    style = Kpoints.supported_modes.Gamma if scheme == "gamma" else Kpoints.supported_modes.Monkhorst
    if abc:
        n_a, n_b, n_c = abc.split(',')
        nkpts = (int(n_a), int(n_b), int(n_c))
        na = 1 if flags[0] else nkpts[0]
        nb = 1 if flags[1] else nkpts[1]
        nc = 1 if flags[2] else nkpts[2]
        kpoints = Kpoints(kpts=((na, nb, nc), ), style=style, comment="kpoints", kpts_shift=shift)
        kpoints.write_file(outfile)
        print(f"KPOINTS file is written to {outfile}.")
    else:
        b_len = poscar.cell.reciprocal().cellpar()
        na = max(1, math.ceil(b_len[0]*2*math.pi/kspacing))
        nb = max(1, math.ceil(b_len[1]*2*math.pi/kspacing))
        nc = max(1, math.ceil(b_len[2]*2*math.pi/kspacing))
        na = 1 if flags[0] else na 
        nb = 1 if flags[1] else nb 
        nc = 1 if flags[2] else nc 
        kpoints = Kpoints(kpts=((na, nb, nc), ), style=style, comment="kpoints", kpts_shift=shift)
        kpoints.write_file(outfile)
        print(f"KPOINTS file is written to {outfile}.")

else:
    pbc      = args.pbc
    nkpoints = args.nkpoints
    kpath    = args.kpath
    pbc_flag = [int(a) for a in pbc.split(',')]
    pbc_flag = [False if a == 0 else 1 for a in pbc_flag ]
    if nkpoints is None:
        bandpath = poscar.cell.bandpath(pbc=pbc_flag)
        #special_points = dict()
        for key in bandpath.special_points.keys():
            #special_points[key] = bandpath.special_points[key].tolist()
            kx, ky, kz = bandpath.special_points[key].tolist()
            kx = Fraction(kx).limit_denominator(100)
            ky = Fraction(ky).limit_denominator(100)
            kz = Fraction(kz).limit_denominator(100)
            print("%-5s%-8s%-8s%-8s"%(key, kx, ky, kz))
        exit(0)

    bandpath = poscar.cell.bandpath(pbc=pbc_flag, eps=0.01)
    if kpath is None:
        labels_list = []
        kpoints_list = []
        for path in bandpath.path.split(','):
            L = list(path)
            labels_list.append(L[0])
            kpoints_list.append(bandpath.special_points[L[0]])

            for i in range(1, len(L)-1):
                labels_list.append(L[i])
                labels_list.append(L[i])
                kpoints_list.append(bandpath.special_points[L[i]])
                kpoints_list.append(bandpath.special_points[L[i]])

            labels_list.append(L[-1])
            kpoints_list.append(bandpath.special_points[L[-1]])
    else:
        labels_list = list(kpath)
        for p in labels_list:
            if not p in bandpath.special_points:
                print(f'Error: point {p} is not a high symmetric point in this structure.')
                exit(0)
        kpoints_list = [bandpath.special_points[p] for p in labels_list]
    
    kpoints = Kpoints(comment="Line mode",
                    style=Kpoints.supported_modes.Line_mode,
                    coord_type="Reciprocal",
                    kpts=kpoints_list,
                    labels=labels_list,
                    num_kpts=nkpoints)
    kpoints.write_file(outfile)
    print(f"KPOINTS file is written to {outfile}.")
