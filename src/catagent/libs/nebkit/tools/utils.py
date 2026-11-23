from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit import rdBase
from ase import Atoms  
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.data import covalent_radii
import numpy as np 
from typing import List, Tuple
from copy import deepcopy

def mol_to_ase_atoms(mol:Chem.Mol):
    new_mol = deepcopy(mol)
    with rdBase.BlockLogs():
        new_mol.UpdatePropertyCache(strict=False)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        AllChem.EmbedMolecule(new_mol, params)
    atomic_numbers = [atom.GetAtomicNum() for atom in new_mol.GetAtoms()] 
    conf = new_mol.GetConformer()
    positions = conf.GetPositions()
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    return atoms 

def get_adsorbate_binding_atom_index(atoms:Atoms) -> tuple | None:
    max_coord = {
        1:(1, 0), #H, 最大配位数1，0对孤对电子
        6:(4, 0), #C

        7:(3, 1), #N，最大配位数3，1对孤对电子
        8:(2, 2), #O
        9:(1, 0), #F

        15:(3, 1), #P
        16:(2, 2), #S
        17: (1, 0), #Cl
    }

    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    numbers = atoms.get_atomic_numbers()

    scores = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        remain_coord = max_coord[numbers[i]][0] - len(indices)
        score = remain_coord if remain_coord > 0 else 0
        if remain_coord < 0:
            score = score + max_coord[numbers[i]][1] + remain_coord * 0.5 
        else:
            score = score + max_coord[numbers[i]][1]
        scores.append(score)
    
    scores = np.array(scores)
    index = np.argsort(scores)

    if scores[index[-1]] == 0: #最大的不饱和度是0，没有键合原子
        return None 
    if len(index) == 1: # 只有一个原子，直接返回 0
        return (0, )
    if scores[index[-2]] == 0: #有一个以上的原子，但第二大的不饱和度是0 
        return (index[-1], )
    return (index[-1], index[-2])
    
        

def find_connected_components(adj_matrix:np.ndarray) -> List[List[int]]:
    '''
    Given an adjacent array of a graph, find the connected components. 
    Args:
        adj_matrix: Adjacent matrix of graph.
    Returns:
        A list of list. Each list contains the indices of node.
    '''
    n = adj_matrix.shape[0]
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
    
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 1:
                union(i, j)

    components_dict = {}
    for i in range(n):
        root = find(i)
        if root not in components_dict:
            components_dict[root] = []
        components_dict[root].append(i)
    
    return list(components_dict.values())


def pattern_to_mol(symbols:str, edges:str):
    symbols_list = symbols.split(",")
    edges_list = edges.split(",")
    mol = Chem.RWMol()
    for s in symbols_list:
        mol.AddAtom(Chem.Atom(s))
    if len(symbols_list) == 1:
        return mol 
    
    for edge in edges_list:
        a,b = edge.split('-')
        mol.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    return mol.GetMol() 

def ase_atoms_to_mol(atoms:Atoms, skin:float=0.1, ignoreH:bool = False) -> Chem.Mol:
    '''
    Convert `ase.Atoms` to `Chem.Mol`. All the bonds are set to be single bond.
    Args:
        atoms: `ase.Atoms` object.
        skin: For atom `A` and `B`, if their distance satisfy $d(A-B) < skin*2 + r(A) + r(B)$, consider them to be bonded. Here r(A) and r(B) are the covalent radii.
        ignoreH: ignore all H atoms.
    '''
    if ignoreH:
        atoms = atoms.copy()
        numbers = atoms.get_atomic_numbers()
        flag = numbers != 1
        atoms = Atoms(numbers=numbers[flag], positions = atoms.positions[flag])

    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True, skin=skin)
    nl.update(atoms)
    natoms = len(atoms)
    mol = Chem.RWMol()
    symbols = atoms.get_chemical_symbols()
    for s in symbols:
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i in range(natoms):
        index, _ = nl.get_neighbors(i)
        for j in index:
            if j > i:
                mol.AddBond(i, int(j), Chem.BondType.SINGLE)
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        AllChem.EmbedMolecule(mol, params)
    return mol.GetMol()

def split_ase_atoms_fragments(atoms:Atoms, skin:float=0.1)->Tuple[List[Atoms], List[List[int]]]:
    '''
    Split `ase.Atoms` into fragments.
    '''
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True, skin=skin)
    nl.update(atoms)
    natoms = len(atoms)
    adjacent_mat = np.zeros((natoms, natoms))
    for i in range(natoms):
        index, _ = nl.get_neighbors(i)
        adjacent_mat[i, index] = 1
    
    components = find_connected_components(adjacent_mat)
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    result = []
    for item in components:
        new_atoms = Atoms(symbols=[symbols[i] for i in item], positions=pos[item])
        result.append(new_atoms)
    return result, components
    


def get_distance_periodic(pos1:np.ndarray, pos2:np.ndarray, cell:np.ndarray):
    if pos1.ndim == 1:
        pos1 = pos1.reshape((1,-1))
    if pos2.ndim == 1:
        pos2 = pos2.reshape((1,-1))
    inv_cell = np.linalg.inv(cell)
    direct1 = pos1 @ inv_cell
    direct2 = pos2 @ inv_cell 
    diff_direct = direct1[:, np.newaxis, :] - direct2[np.newaxis, :, :]
    diff_direct = diff_direct - np.round(diff_direct)
    diff_cart = diff_direct @ cell 
    dist_cart = np.sqrt(np.sum(diff_cart**2, axis=2)) 
    return dist_cart


def get_structures_distance(atoms1:Atoms, atoms2:Atoms)->float:
    '''
    Return the distance between two structures. Consider the periodic boundary condition.
    The number of atoms and cell must match in atoms1 and atoms2, otherwise it return -1.0.

    Args:
        atoms1: The ase.Atoms object of the first structure.
        atoms2: The ase.Atoms object of the second structure.
    
    Returns:
        The distance between two structures.
    '''
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



def remove_duplicated_images(images:List[Atoms], threshold:float):
    nimg = len(images)
    if nimg <= 3:
        return images 
    
    delete_index = []
    ref_index = 0
    for i in range(1, nimg):
        d = get_structures_distance(images[ref_index], images[i])
        if d < threshold:
            delete_index.append(i)
        else:
            ref_index = i 
    new_images = [images[j] for j in range(nimg) if not j in delete_index]
    return new_images 

