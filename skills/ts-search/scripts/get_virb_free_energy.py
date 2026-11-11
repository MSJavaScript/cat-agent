"""
Calculate vibration free energy from the result of frequency analysis (OUTCAR).
Usage:
python3 get_virb_free_energy.py --outcar OUTCAR -t 300

# Use --isotope option to set the atomic mass of isotope
# For each tuple, first element is the atom index
# second element is the atomic mass
python3 get_virb_free_energy.py --outcar OUTCAR -t 300 --isotope "(20,2.0), (21, 2.0)"
"""
import numpy as np
from subprocess import check_output
import argparse, re
from io import StringIO


EV = 1.602176634e-19            # J / eV
AMU = 1.66053906660e-27         # kg / amu
ANG2 = 1e-20                    # m^2 / A^2
C_CM = 2.99792458e10            # cm / s
KB = 8.617333262e-5             # eV / K
HC = 1.239841984e-4             # eV * cm (energy of a 1 cm^-1 photon)

def vibration(T, wavenums):
    if (len(wavenums) == 0):
        return 0
    C = 96.4853240536413
    #A = 100*C*h/k_b
    A = 1.43877735382772
    #B = R/1000
    B = 8.31446261815324*0.001
    logZ = 0.0
    for w in wavenums:
        if (w < 0.01):
            continue
        logZ += ((-w*A)/(2*T) - np.log(1 - np.exp(-w*A/T)))
    return -logZ*T*B/C

parser = argparse.ArgumentParser(description="Calculate the vibration free energy.")
parser.add_argument("--outcar", type=str, required=True, help="Path to the OUTCAR of frequency analysis.")
parser.add_argument("-t", "--temperature", type=float, required=True, help="Temperature in Kelvins (K).")
parser.add_argument("--isotope", type=str, help="Relative atomic masses of isotope, set by a list of tuples. For example, '(20, 2.0),(21, 3.0),(23, 2.0)' . Where the first element is atomic index, start from 0, second element is relative atomic masses.")
args = parser.parse_args()

if args.isotope is None:
    command = f"grep cm-1 {args.outcar} | grep -v 'f/i' | awk '{{print $8}}'"
    wavenums = check_output(command, shell=True) 
    wavenums = np.array([float(item.decode('utf-8').strip()) for item in wavenums.split()])
else:
    command = f"grep 'ions per type' {args.outcar} | awk -F= '{{print $2}}'"
    nions_per_type = check_output(command, shell=True).decode("utf-8")
    nions_per_type = list(map(int, nions_per_type.split()))
    nions_total = sum(nions_per_type)

    atom_mass = dict()
    for item in re.findall(r'\([^)]*\)', args.isotope):
        a = item.replace('(', '').replace(')', '').replace(' ', '')
        k, v = a.split(',')
        k = int(k)
        if k >= nions_total:
            print(f"Error: Atom index {k} should be less than the total number of atoms {nions_total}.")
            exit(1)
        atom_mass[k] = float(v)

    command = f"grep POMASS {args.outcar} | tail -1 | awk -F= '{{print $2}}'"
    pomass = check_output(command, shell=True).decode("utf-8")
    pomass = list(map(float, pomass.split()))

    mass_array = []
    for nions, mass in zip(nions_per_type, pomass):
        mass_array.extend([mass]*nions)
    for k in atom_mass.keys():
        mass_array[k] = atom_mass[k]
    mass_array = np.array(mass_array)

    command = f"grep -n 'SECOND DERIVATIVES' {args.outcar} | awk -F: '{{print $1}}'"
    a = check_output(command, shell=True)
    a = int(a.decode('utf-8').strip())

    command = f"sed -n '{a+2}p' {args.outcar}"
    header = check_output(command, shell=True)

    # header of second derivative table, has the format
    # 17X', '17Y', '17Z', '18X', '18Y', '18Z' ...
    header = [item.decode('utf-8').strip() for item in header.split()]
    mass = np.array([mass_array[int(c[:-1])-1] for c in header])

    nline = len(header)
    command = f"sed -n '{a+3}, {a+2+nline}p' {args.outcar} | awk '{{$1=\"\"; print}}'"
    second_derivative = check_output(command, shell=True)

    # second derivative (not symmetrized)
    second_derivative = np.loadtxt(StringIO(second_derivative.decode("utf-8")))
    hessian = -0.5 * (second_derivative + second_derivative.T)

    dyn = hessian / np.sqrt(np.outer(mass, mass))
    eigenvalues = np.linalg.eigvalsh(dyn)

    # # eigenvalue of the mass weighted Hessian (eV / (A^2 amu)) -> wavenumber (cm^-1)
    # W2_TO_CM = np.sqrt(EV / (ANG2 * AMU)) / (2 * np.pi * C_CM)
    W2_TO_CM = 521.4708983725066
    wavenums = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * W2_TO_CM

#print("Free energy at %.2f K: %-.3f eV"%(args.temperature, vibration(args.temperature, wavenums)))
wavenums = wavenums[wavenums > 0]
energies = HC * wavenums                       # eV per vibrational quantum
zpe = 0.5 * energies.sum()

if args.temperature > 0.0:
    x = np.clip(energies / (KB * args.temperature), None, 700.0)
    em1 = np.expm1(x)
    u = (0.5 * energies + energies / em1).sum()
    s = KB * (x / em1 - np.log(1.0 - np.exp(-x))).sum()
    f = zpe + KB * args.temperature * np.log(1.0 - np.exp(-x)).sum()
else:
    u, s, f = zpe, 0.0, zpe

print("Temperature: %.2f K, ZPE: %-.3f eV, Free energy: %-.3f eV, Entropy: %-.3e eV/K."%(args.temperature, zpe, f, s))





