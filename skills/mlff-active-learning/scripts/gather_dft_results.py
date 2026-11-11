"""
Given a source dir, gather the DFT energies and forces from OUTCAR.

Usage:
python3 gather_dft_results.py --root_dir "./" -o dataset1.extxyz --first 5
"""

from ase.io import read, write 
import argparse, re
from pathlib import Path
from random import sample 
import numpy as np 
from subprocess import check_output

def get_structures_distance(atoms1, atoms2)->float:
    if len(atoms1)!= len(atoms2):
        return np.nan
    cell1 = atoms1.get_cell()
    cell2 = atoms2.get_cell()
    if not np.all(np.isclose(cell1, cell2)):
        return np.nan    
    frac1 = atoms1.get_scaled_positions()
    frac2 = atoms2.get_scaled_positions()
    diff_frac = frac1 - frac2
    diff_frac[diff_frac > 0.5] -= 1.0
    diff_frac[diff_frac < -0.5] += 1.0
    diff_cart = np.matmul(diff_frac, cell1)
    return np.sum(diff_cart**2.0) ** 0.5 

def connected_components(flag):
    n = len(flag)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        # BFS 收集当前连通分量
        comp = []
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if flag[u][v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(sorted(comp))
    return components

parser = argparse.ArgumentParser(description="Gather the DFT energies and forces from VASP OUTCARs.")
parser.add_argument("--root_dir", type=str, default="./", help="Root dir of VASP results. Default is the current directory.")
parser.add_argument("-o", "--outfile", type=str, required=True, help="The name of extxyz file to save the configurations.")
parser.add_argument("--min_dist", type=float, default=0.2, help="If the distance between two structures is less than this value, they are taken as the same. Default 0.2.")
parser.add_argument("--first", type=int, default=5, help="Select first n configurations from the OUTCAR.")
parser.add_argument("--last", type=int, help="Select last n configurations from the OUTCAR.")
parser.add_argument("--random", type=int, help="Randomly select n configurations from the OUTCAR.")

args = parser.parse_args()
collect_configs = []
n_first = args.first 
n_last = args.last if args.last else -1 
n_random = args.random if args.random else -1 
min_dist = args.min_dist if args.min_dist > 0.2 else 0.2

for outcar in Path(args.root_dir).rglob("OUTCAR*"):
    try:
        configs = read(outcar, index=":", format="vasp-out")
        for config in configs:
            config.set_constraint([])
    except Exception as e:
        continue

    # 从中去掉没有收敛的DFT
    output = check_output(f"grep -E 'total energy-change | aborting loop' {outcar}", shell=True).decode("utf-8")
    Ediff = check_output(f"grep 'stopping-criterion for ELM' {outcar} | awk '{{print $3}}'", shell=True).decode("utf-8")
    Ediff = float(Ediff.strip())

    dE_list = []
    dE = None 
    for line in output.split('\n'):
        if "energy-change" in line:
            m = re.search(r":([^(]*)\(", line)
            if m:
                dE = float(m.group(1).strip())
        elif "aborting" in line:
            dE_list.append(dE)
            dE = None 

    nconfigs = len(configs)
    if len(dE_list) != nconfigs:
        print(f"WARNNING: The number of structures and the number of energy changes are not the same in {outcar}. Skipping it.")
        continue

    configs_2 = [configs[i] for i in range(nconfigs) if abs(dE_list[i]) < Ediff]
    nconfigs = len(configs_2)
    configs = configs_2 

    indices_set = set()
    indices_set.update(range(n_first))
    if n_last > 0:
        indices_set.update(range((nconfigs-n_last), nconfigs))
    if 0 < n_random < nconfigs:
        indices_set.update(sample(range(nconfigs), n_random))
    indices = [i for i in indices_set if i >= 0 and i < nconfigs]
    indices.sort()
    if len(indices) > 0:
        for i in indices:
            collect_configs.append(configs[i])

nconfigs = len(collect_configs)
dist_mat = np.zeros((nconfigs, nconfigs))
for i in range(nconfigs):
    for j in range(i+1, nconfigs):
        dist_mat[i,j] = get_structures_distance(collect_configs[i], collect_configs[j])
        dist_mat[j,i] = dist_mat[i,j]

flag = dist_mat < min_dist
np.fill_diagonal(flag, False)
components = connected_components(flag)
result_configs = []
for comp in components:
    result_configs.append(collect_configs[comp[0]])

print(f"{len(result_configs)} structures are gathered and written to {args.outfile}.")
write(args.outfile, result_configs, format="extxyz")
