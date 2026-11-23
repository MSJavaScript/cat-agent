from typing import List
import numpy as np 

from ase.data import covalent_radii
from ase import Atoms 
from ase.calculators.calculator import Calculator, all_changes

from .tools import rigid_body_forces


class SurfaceAttractor(Calculator):
    '''
    Add LJ potential between atoms `adsorbate_atom_index` and `attraction_atoms_index` to attract these atoms to the surface.
    '''
    def __init__(self, 
                adsorbate_atom_index:List[int],
                attraction_atoms_index:List[int],
                atomic_numbers:List[int],
                scale_factor:float = 0.05,
                **kwargs):
        '''
        Args:
            adsorbate_atom_index: Indices of adsorbate atoms
            attraction_atoms_index: Indices of surface atoms which attract the adsorbate atoms.
            atomic_numbers: List of atomic numbers.
            scale_factor: Scale factor to scale the force on H atoms.
        '''
        Calculator.__init__(self, **kwargs)

        a = set(adsorbate_atom_index)
        b = set(attraction_atoms_index)
        if not a.isdisjoint(b):
            raise ValueError("There can not be common atomic indices in `adsorbate_atom_index` and `attraction_atoms_index`.")
        
        self.adsorbate_atom_index = list(a)
        self.attraction_atoms_index = list(b)
        self.remain_index = np.setdiff1d(np.arange(len(atomic_numbers)), self.adsorbate_atom_index).tolist()
        self.scale_force = True if len(adsorbate_atom_index) > 1 else False 
        self.scale_factor = scale_factor
        self.implemented_properties = ["energy", "forces"]
    
    def calculate(self,
                  atoms:Atoms=None,
                  properties=["energy", "forces"],
                  system_changes=all_changes):

        super().calculate(atoms)

        natoms = len(atoms)
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        numbers = atoms.get_atomic_numbers()
        inv_cell = np.linalg.inv(cell)

        pos_ads = pos[self.adsorbate_atom_index]
        pos_remain = pos[self.remain_index]

        direct_ads = pos_ads @ inv_cell
        direct_remain = pos_remain @ inv_cell  
        diff_direct = direct_ads[:, np.newaxis, :] - direct_remain[np.newaxis, :, :]
        diff_direct = diff_direct - np.round(diff_direct)
        diff_cart = diff_direct @ cell 
        dist_cart = np.sqrt(np.sum(diff_cart**2, axis=2)) #(N_rigid, N_attra)

        E = 0
        forces = np.zeros((natoms, 3))
        aa = 2 ** (1.0/6)
        for i, idx in enumerate(self.adsorbate_atom_index):
            ri = covalent_radii[numbers[idx]]
            f = np.array([0, 0, 0], dtype=float)
            for j, jdx in enumerate(self.remain_index):
                rj = covalent_radii[numbers[jdx]]
                sigma = ri + rj 

                if dist_cart[i, j] > aa*sigma: 
                    # 当原子间距离大于取极小值的距离时，如果jdx不再吸引原子的索引中则跳过
                    if not jdx in self.attraction_atoms_index:
                        continue
                    else: #如果在吸引原子的索引中，则替换成二次函数势，防止LJ势在太远时变成0
                        dr = dist_cart[i, j] - aa*sigma
                        E += dr**2
                        f += -2*dr*diff_cart[i, j]/dist_cart[i, j]
                        continue 

                d = dist_cart[i, j] + 0.01
                a = sigma / d 
                a6 = a**6
                E += 4*a6*(a6-1)
                f += 24*a6*(2*a6-1)/d*diff_cart[i, j]

            forces[idx] = f*self.scale_factor if self.scale_force and numbers[idx] == 1 else f 
        
        forces[self.adsorbate_atom_index] = rigid_body_forces(pos_ads, forces[self.adsorbate_atom_index])
        self.results = {}
        self.results["energy"] = E
        self.results["forces"] = forces
