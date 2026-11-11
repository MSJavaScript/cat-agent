from ase import Atoms 
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import MDMin
from ase.mep import NEB 
from ase.io import write 

from typing import List, Tuple

from libs.nebkit.tools.utils import split_ase_atoms_fragments, get_distance_periodic, get_structures_distance, remove_duplicated_images
from libs.nebkit.calculator.surface_attractor import SurfaceAttractor
from libs.nebkit.calculator.tools import adjust_positions
from libs.nebkit.calculator.idpp import idpp_interpolate

import numpy as np



def _adjust_intermediate_image(atoms:Atoms, 
                               adsorbate_atom_index:List[int],
                               threshold:float = 0.3,
                               skin:float = 0.2):
    '''
    This helper function move the adsorbate atoms to the surface, to avoid unreasonable structures. What does it do:
    '''
    cell = atoms.get_cell()
    adsorbate_atoms = Atoms(numbers = atoms.numbers[adsorbate_atom_index],
                            positions = atoms.positions[adsorbate_atom_index])
    fragments, mapping_list = split_ase_atoms_fragments(adsorbate_atoms)
    atomic_numbers = atoms.get_atomic_numbers()
    nfragments = len(fragments)
    if nfragments != 2:
        return 
    
    center = []
    diameter = []
    for frag in fragments:
        frag.set_cell(cell)
        numbers = frag.get_atomic_numbers()
        if len(numbers) == 1:
            center.append(np.array([frag.positions[0,0],frag.positions[0,1],frag.positions[0,2]]))
            diameter.append(covalent_radii[numbers[0]])
            continue
        center.append(np.mean(frag.positions, axis=0))
        dist_mat = frag.get_all_distances(mic=True)
        i1, i2 = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
        diameter.append(dist_mat[i1, i2] + covalent_radii[numbers[i1]] + covalent_radii[numbers[i2]])
    
    dist = get_distance_periodic(center[0], center[1], cell)
    dist = dist[0,0] - diameter[0] - diameter[1]
    if dist < threshold: #如果是一个整体同样返回不需要调整
        return 
    
    # 检测哪个分子片没有与表面接触
    dist_mat = atoms.get_all_distances(mic=True) 
    nl = NeighborList(cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True, skin=skin)
    nl.update(atoms)

    for li in mapping_list:
        adj_list = []
        fragment_index = [adsorbate_atom_index[i] for i in li]
        for i in fragment_index:
            index, _ = nl.get_neighbors(i)
            for j in index:
                if not j in adsorbate_atom_index:
                    adj_list.append(j)
        if len(adj_list) != 0:
            continue
        for i in fragment_index:
            sort_index = np.argsort(dist_mat[i])
            a = []
            for j in sort_index:
                if not j in adsorbate_atom_index:
                    a.append(j)
                if len(a) > 1:
                    break 
            adj_list.extend(a)
        adj_list = np.unique(np.array(adj_list))
        atoms.calc = SurfaceAttractor(adsorbate_atom_index=fragment_index,
                                      attraction_atoms_index=adj_list,
                                      atomic_numbers = atomic_numbers)
        
        opt = MDMin(atoms=atoms, logfile='temp.log')
        opt.run(fmax=0.2, steps=50)
    
    return atoms 
        
    

def _adjust_initial_final_state(
        ini_state:Atoms, 
        fin_state:Atoms,
        adsorbate_atom_index: List[int])-> None | Tuple[List[Atoms], bool]:
    
    '''
    A helper function to move physically adsorbated initial or final states to the surface.
    Args:
        ini_state: Initial state.
        fin_state: Final state.
        adsorbate_atom_index: The atom indices of adsorbate atoms.
    '''
    mol_index_set = set(adsorbate_atom_index)

    nl = NeighborList(cutoffs=natural_cutoffs(ini_state), bothways=True, self_interaction=False)
    nl.update(ini_state)
    ini_bond_atom_index = []
    for ii in mol_index_set:
        indices, offsets = nl.get_neighbors(ii)
        ini_bond_atom_index.extend(set(indices.tolist()).difference(mol_index_set))
    
    nl = NeighborList(cutoffs=natural_cutoffs(fin_state), bothways=True, self_interaction=False)
    nl.update(fin_state)
    fin_bond_atom_index = []
    for ii in mol_index_set:
        indices, offsets = nl.get_neighbors(ii)
        fin_bond_atom_index.extend(set(indices.tolist()).difference(mol_index_set))
    
    ini_n_bond_atoms = len(ini_bond_atom_index)
    fin_n_bond_atoms = len(fin_bond_atom_index)

    if ini_n_bond_atoms == 0 and fin_n_bond_atoms == 0: 
        return # 初态和末态都是物理吸附，什么都不做
    if ini_n_bond_atoms > 0 and fin_n_bond_atoms > 0:
        return # 初态和末态都是化学吸附，什么都不做
    
    ads_pos_ini = ini_state.positions[adsorbate_atom_index]
    ads_pos_fin = fin_state.positions[adsorbate_atom_index]
    if ini_n_bond_atoms > 0: #末态是物理吸附 
        ini_state_moved = False 
        dpos = ads_pos_ini - ads_pos_fin
        new_fin_state = fin_state.copy()
        new_fin_state.positions[adsorbate_atom_index] = adjust_positions(ads_pos_fin, dpos)
        images = [new_fin_state, new_fin_state.copy(), fin_state]
    else: 
        ini_state_moved = True
        dpos = ads_pos_fin - ads_pos_ini
        new_ini_state = ini_state.copy()
        new_ini_state.positions[adsorbate_atom_index] = adjust_positions(ads_pos_ini, dpos)
        images = [ini_state, new_ini_state.copy(), new_ini_state]
    
    neb = NEB(images, method='improvedtangent')
    neb.interpolate(method='linear')
    images = idpp_interpolate(neb.images, adsorbate_atom_index, True)
    return images, ini_state_moved

def build_surface_neb_path(ini_state:Atoms,
                           fin_state:Atoms,
                           adsorbate_atom_index:List[int],
                           image_spacing:float = 0.5,
                           n_images:int = None) -> List[Atoms]:
    '''
    Given the initial and final states of a **surface reaction**, this function constructs the initial NEB path, Two special cases are handled: 
    - Moving physically adsorbed molecules closer to the surface.
    - Adjust the intermediate images of coupling or dissociation reactions, if the fragments do not make contact with the surface.
    Args:
        ini_state: `ase.Atoms` object of initial state.
        fin_state: `ase.Atoms` object of final state.
        adsorbate_atom_index: Atomic indices of adsorbate atoms.
        image_spacing: Spacing between adjacent images. This argument is ignored if `n_images` is not `None`.
        n_images: The number of interpolation images.
    '''
    n_insert = 2
    extra = _adjust_initial_final_state(ini_state, fin_state, adsorbate_atom_index)
    if extra is None:
        new_ini_state = ini_state.copy()
        new_fin_state = fin_state.copy()
        ini_state_moved = None 
    else:
        extra_images, ini_state_moved = extra
        if ini_state_moved:
            new_ini_state = extra_images[n_insert]
            new_fin_state = fin_state.copy()
        else:
            new_ini_state = ini_state.copy()
            new_fin_state = extra_images[0]

    if n_images is None:
        # 计算初末态之间的距离
        d = get_structures_distance(new_ini_state, new_fin_state)
        n_images = int(d / image_spacing) - 1

    images = [new_ini_state]
    images += [new_ini_state.copy() for i in range(n_images)]
    images.append(new_fin_state)
    neb = NEB(images, method='improvedtangent')
    neb.interpolate(method='linear')
    images = idpp_interpolate(neb.images, adsorbate_atom_index, True)

    # 对中间态的image进行调节
    for i in range(1, len(images)-1):
        img = _adjust_intermediate_image(images[i], adsorbate_atom_index, threshold=0.2, skin=0.15)
        if not img is None:
            images[i] = img 
    
    if not ini_state_moved is None:
        if ini_state_moved:
            images = extra_images[0:n_insert] + images 
        else:
            images = images + extra_images[1:(n_insert+1)]

    images = remove_duplicated_images(images, image_spacing*0.5) 
    return images 
    