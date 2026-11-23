"""
Extract the vibration mode from OUTCAR.

Usage:
# Only print the wave numbers of vibration mode
python3 get_virb_mode.py --outcar OUTCAR -p

# Extract the imaginary vibration mode with highest energy
# can be used as MODECAR if NEB results does not exist
python3 get_virb_mode.py --outcar OUTCAR

# Extract a specific vibration mode
python3 get_virb_mode.py --outcar OUTCAR -e 10
"""

import numpy as np
from subprocess import check_output, check_call
import argparse

parser = argparse.ArgumentParser(description="Extract the vibration mode from OUTCAR.")
parser.add_argument("--outcar", type=str, required=True, help="Path to the OUTCAR of frequency analysis.")
parser.add_argument("-p", "--print_wavenum", action="store_true", help="Only print wave numbers of vibration mode. Wave number of imaginary frequency is negative.")
parser.add_argument("-e", "--extract", type=int, help="Extract the vibration mode with index 'i' (start from 0) to file 'mode_i'. By default, extract the imaginary mode with the highest energy. This can be used as MODECAR for dimer calculation.")

args = parser.parse_args()

command = f"grep cm-1 {args.outcar} | awk -F'cm-1' '{{print $1}}' | awk -F'2PiTHz' '{{print $2}}'"
wavenums = check_output(command, shell=True) 
wavenums = np.array([float(item.decode('utf-8').strip()) for item in wavenums.split()])

command = f"grep cm-1 {args.outcar} | awk -F= '{{print $1}}' | awk '{{print $2}}'"
flags = check_output(command, shell=True)
flags = [item.decode('utf-8').strip() for item in flags.split()]
signs = np.array([1 if f == 'f' else -1 for f in flags])
wavenums = wavenums * signs

if args.print_wavenum:
    print("Wave numbers (cm-1):")
    for i, a in enumerate(wavenums):
        print("%-12.2f"%(a), end='')
        if (i+1)%5 == 0:
            print("")
    print("")
    exit(0)

i_max = -1
i_w = 0
for i in range(len(wavenums)):
    if wavenums[i] < i_w:
        i_w = wavenums[i]
        i_max = i

command = f"grep NIONS {args.outcar} | awk '{{print $12}}'"
natoms = check_output(command, shell=True)
natoms = int(natoms.decode('utf-8').strip())

command = f"grep -n cm-1 {args.outcar} | awk -F: '{{print $1}}'"
linenum = check_output(command, shell=True)
linenum = [int(item.decode('utf-8').strip()) for item in linenum.split()]
nmode = len(linenum)

if args.extract is None:
    if i_max == -1:
        print("Error: '-e' option is not set and there is no imaginary mode.")
        exit(1)
    else:
        print("Warnning: '-e' option is not set, extract imaginary mode with highest energy by default.")
        idx = i_max 

if args.extract:
    if args.extract >= nmode:
        print(f"Error: '-e' option {args.extract} should be less than the number of modes {nmode}.")
        exit(1)
    else:
        idx = args.extract

print(f"Mode {idx} is save to mode_{idx}.")
command = f"sed -n '{linenum[idx]+2},{linenum[idx]+1+natoms} p' {args.outcar} | awk '{{printf \"%-12.6f%-12.6f%-12.6f\\n\", $4, $5, $6}}' > mode_{idx}"
check_call(command, shell=True)

