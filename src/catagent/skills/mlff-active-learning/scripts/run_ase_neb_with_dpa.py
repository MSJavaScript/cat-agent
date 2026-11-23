"""
Run structure relaxation or NEB with DPA force field.
Usage:

# run structure relaxation
python3 run_ase_neb_with_dpa.py \
  --pt fronzen.pt2 \
  --save_name opt1 \
  opt --infile POSCAR

# Add constraint on the bond length between atom 10 and 12. 
python3 run_ase_neb_with_dpa.py \
  --pt fronzen.pt2 \
  --save_name opt2 \
  opt --infile POSCAR --bond "10,12"

# Set the atomic indices of adsorbate, check if the adsorbate dissociated or combined 
python3 run_ase_neb_with_dpa.py \
  --pt fronzen.pt2 \
  --save_name opt3 \
  opt --infile POSCAR --ads_indices "10,11,12,13"

# Run NEB, providing initial and final states
python3 run_ase_neb_with_dpa.py \
  --pt fronzen.pt2 \
  --save_name neb1 \
  neb --neb_ini POSCAR_ini --neb_fin POSCAR_fin

# Run NEB, providing an initial NEB path
python3 run_ase_neb_with_dpa.py \
  --pt fronzen.pt2 \
  --save_name neb2 \
  neb --neb_path neb_path.extxyz

"""

from ase.io import read, write 
from ase.optimize import FIRE, LBFGS
from ase.mep import NEB
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixBondLength

from subprocess import check_call
from deepmd.calculator import DP as DPCalculator
import argparse, json
import numpy as np

def run_opt(atoms, 
            pt_file, 
            save_name, 
            fmax=0.08, 
            nsteps=300,
            atom_pair:tuple=None,
            ads_index:list =None):
    
    result = dict()    
    if ads_index:
        nads_atoms = len(ads_index)
        idx_dict = {ads_index[i]:i for i in range(len(ads_index))} 
        adj_mat = np.zeros((nads_atoms, nads_atoms))
        nl = NeighborList(cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        for ii in ads_index:
            indices, offsets = nl.get_neighbors(ii)
            for jj in indices:
                if jj in ads_index:
                    adj_mat[idx_dict[ii], idx_dict[jj]] = 1
                    adj_mat[idx_dict[jj], idx_dict[ii]] = 1

    atoms.calc = DPCalculator(pt_file)
    if atom_pair:
        c = FixBondLength(atom_pair[0], atom_pair[1])
        atoms.set_constraint(atoms.constraints + [c]) 
    try:
        opt = LBFGS(atoms, logfile=f"{save_name}_opt.log", trajectory=f"{save_name}_opt.traj")
        opt.run(fmax=fmax, steps=nsteps)
        if not opt.converged():
            result = {"state": "error", "message": f"Cannot converge after {nsteps} steps."}
    except Exception as e:
        if atom_pair:
            result = {"state": "error", "message": f"Cannot converge with FixBondLength{atom_pair} constraint."} 
        else:
            result = {"state": "error", "message": f"{e}"}

    # 优化之后吸附物原子之间的邻接矩阵
    if ads_index:
        new_adj_mat = np.zeros((nads_atoms, nads_atoms))
        nl = NeighborList(cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        for ii in ads_index:
            indices, offsets = nl.get_neighbors(ii)
            for jj in indices:
                if jj in ads_index:
                    new_adj_mat[idx_dict[ii], idx_dict[jj]] = 1
                    new_adj_mat[idx_dict[jj], idx_dict[ii]] = 1
        if not np.all(new_adj_mat == adj_mat):
            result = {"state": "error", "message": "The adjacent matrix of adsorbate changed after relaxtion."}

    result = {"state": "success", "message": "Converged."} if not result else result
    with open(f"{save_name}_opt_state.json", mode='w') as f:
        json.dump(result, f)

    write(f"POSCAR_{save_name}.result", atoms, format="vasp")
    
    check_call(f"rm -f {save_name}_opt.log", shell=True)
    
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

def run_neb(pt_file, 
            save_name,
            atoms_ini=None, 
            atoms_fin=None, 
            spacing=0.5, 
            images=None,
            fmax=0.08, 
            nsteps=300):
    if images:
        for item in images:
            item.calc = DPCalculator(pt_file)
        neb = NEB(images, method='improvedtangent', k=0.1, climb=False)
    else:
        d = get_structures_distance(atoms_ini, atoms_fin)
        nimages = int(d / spacing) - 1

        images = [atoms_ini]
        images += [atoms_ini.copy() for _ in range(nimages)]
        images.append(atoms_fin)

        for item in images:
            item.calc = DPCalculator(pt_file)

        neb = NEB(images, method='improvedtangent', k=0.1, climb=False)
        neb.interpolate(method='idpp', mic=True) 

    opt1 = FIRE(neb, maxstep=0.05,
                logfile=f"{save_name}_prev_neb.log", 
                trajectory=f"{save_name}_prev_neb.traj")
    opt1.run(fmax=fmax, steps=50)

    neb2 = NEB(neb.images, method='improvedtangent', k=0.1, climb=True)
    opt2 = FIRE(neb2, maxstep=0.1, 
                logfile=f"{save_name}_neb.log", 
                trajectory=f"{save_name}_neb.traj")
    opt2.run(fmax=fmax, steps=nsteps)

    if not opt2.converged():
        result = {"state": "error", "message": f"Cannot converge after {nsteps} steps."}
    else:
        result = {"state": "success", "message": "Converged."}

    with open(f"{save_name}_neb_state.json", mode='w') as f:
            json.dump(result, f)

    write(f"{save_name}.result.extxyz", 
          neb2.images, 
          format="extxyz")
    
    check_call(f"rm -f {save_name}_prev_neb.log  {save_name}_prev_neb.traj {save_name}_neb.log", shell=True)


parser = argparse.ArgumentParser(description="Use dpa4 force field run structure relaxation and NEB. Cell parameters are not relaxed.")

parser.add_argument("--pt", 
                    required=True, 
                    type=str, 
                    help="Path to DPA checkpoint file.")

parser.add_argument("--save_name", 
                    required=True, 
                    type=str, 
                    help="Prefix of the *.traj and *.json file names.")

parser.add_argument("--fmax", type=float, default=0.08, help="Force convergence criterion (eV/A). Default 0.08. ")
parser.add_argument("--nsteps", type=int, default=350, help="Max relaxation or NEB steps. Default 350.")

subparsers = parser.add_subparsers(dest='task', help='Task type, run structure relaxation or NEB.')

opt_parser = subparsers.add_parser('opt', help='Structure relaxation')
opt_parser.add_argument("--infile", 
                        type=str,
                        help="Path to the input POSCAR file which requires structure relaxation.")
opt_parser.add_argument("--bond", 
                        type=str,
                        help="Two integers separated by comma, FixBondLength constraint will be applied to these two atoms.")
opt_parser.add_argument("--ads_indices", 
                        type=str,
                        help="Indices of adsorbate atoms, a list of integers separated by comma.")

neb_parser = subparsers.add_parser('neb', help='NEB task.')
neb_parser.add_argument("--neb_ini",
                    type=str,
                    help="Path to initial state.")
neb_parser.add_argument("--neb_fin",
                    type=str,
                    help="Path to final state.")
neb_parser.add_argument("--neb_path",
                    type=str,
                    help="Path to an extxyz file which contains an initial NEB path.")

args = parser.parse_args()

pt_file   = args.pt 
save_name = args.save_name

if args.task == 'opt':
    infile    = args.infile
    bond      = args.bond 
    ads_indices = args.ads_indices
    if bond:
        a, b = bond.replace(' ', '').split(',')
        atoms_pair = (int(a), int(b))
    else:
        atoms_pair = None 

    if ads_indices:
        ads_indices = [int(i) for i in ads_indices.replace(' ', '').split(',')]
    else:
        ads_indices = None 

    run_opt(read(infile), pt_file, save_name, args.fmax, args.nsteps, atoms_pair, ads_indices)
else:
    neb_ini   = args.neb_ini
    neb_fin   = args.neb_fin
    neb_path  = args.neb_path
    if neb_path:
        images = read(neb_path, index=":")
        run_neb(pt_file, save_name, None, None, 0.5, images, args.fmax, args.nsteps)
    else:
        if neb_ini is None or neb_fin is None:
            print("Error: Must set initial and final states structures if you don't provide a NEB path.")
            exit(0)
        atoms_ini = read(neb_ini)
        atoms_fin = read(neb_fin)
        run_neb(pt_file, save_name, atoms_ini, atoms_fin, 0.5, None,  args.fmax, args.nsteps)