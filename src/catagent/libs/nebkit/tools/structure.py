from ase import Atoms 
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from typing import List
import numpy as np

def sort_elements(atoms:Atoms)->Atoms:
    '''
    When you use ase to repeat a structure, the structure is simply tiled one by one, the same elements are not grouped together. This function sort the elements and keep the constraints (fixed atoms) in the system.
    
    Args:
        atoms: The structure needs to sort.
    '''
    pos = atoms.get_positions()
    natoms = pos.shape[0]
    atoms_count = atoms.symbols.formula.count()
    index_count = {key:[] for key in atoms_count.keys()}
    symbols = atoms.get_chemical_symbols()
    for i, s in enumerate(symbols):
        index_count[s].append(i)

    numbers = []
    index = []
    for key in index_count.keys():
        numbers.extend([atomic_numbers[key]] * len(index_count[key]))
        index.extend(index_count[key])

    constraints = None
    if len(atoms.constraints) != 0:
        fix_indices = atoms.constraints[0].get_indices()
        flag = np.zeros((natoms, ), dtype=bool)
        flag[fix_indices] = True
        new_flag = flag[index]
        new_fix_indices = np.where(new_flag)[0].tolist()
        constraints = FixAtoms(indices=new_fix_indices)

    new_pos = pos[index, :]
    new_atoms = Atoms(numbers = numbers,
                    positions = new_pos,
                    cell = atoms.get_cell())

    if not constraints is None:
        new_atoms.set_constraint(constraints) 
    
    return new_atoms

def group_surface_layers(surface:Atoms, threshold:float = 0.25)->List[np.ndarray]: 
    '''
    Group the atoms by coordinates along the normal vector of plane ab (the hkl plane).

    Args:
        surface: An ase.Atoms object which represent a slab.
        threshold: If the difference in coordinates between atoms is less than this threshold, these atoms are classified into the same layer.

    Returns:
        A list, in which each item is a list of atom index.
    '''

    vec_a = surface.cell[0]
    vec_b = surface.cell[1]
    vec_n = np.cross(vec_a, vec_b)
    vec_n = vec_n / np.linalg.norm(vec_n)
    z_n = np.zeros((len(surface), ))
    for i, p in enumerate(surface.positions):
        z_n[i] = np.dot(vec_n, p)
    min_z_n = np.min(z_n) - 0.1
    max_z_n = np.max(z_n) + 0.1
    bins = np.linspace(min_z_n, max_z_n, int((max_z_n - min_z_n)/threshold))
    bin_indices = np.digitize(z_n, bins, right=False)
    result = []
    for i in range(1, len(bins)):
        indices = np.where(bin_indices == i)[0]
        if len(indices) != 0:
            result.append(indices)
    return result 