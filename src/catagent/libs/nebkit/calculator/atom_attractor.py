from typing import List,Tuple 
import numpy as np 

from ase.data import covalent_radii
from ase import Atoms 
from ase.calculators.calculator import Calculator, all_changes

from itertools import product

from .tools import rigid_body_forces

class AtomAttractor(Calculator):
    '''
    这个类用于把初态和末态摆放到表面上。ReactionHelper会主动把原子编号排序同时进行原子对齐（例如对于CO2解离反应，如何CO2相对于解离后的CO+O旋转了180°，ReactionHelper会纠正过来）。不能把反应物和产物分开，或把分子片段拆开分别往表面上放，因为这会破坏已有的原子序号或对齐的原子位置。所以这里选择的做法是，把反应物和产物维持在组合的状态，分别找它们的键合原子和吸附位点，把整个反应物和产物都吸引过去。
    '''
    def __init__(self, 
                nsurf_atoms:int,
                atom_group_pairs: List[Tuple[list, list]],
                **kwargs):
        Calculator.__init__(self, **kwargs)

        # 这里总是假设表面原子在前，吸附物原子在后
        self.nsurf_atoms = nsurf_atoms

        # 每个tuple的第一个元素是吸附物中原子的索引（单独在吸附物中的，不是在表面+吸附物这个整体中的）
        # 第二个元素是表面原子的索引
        all_pairs = []
        for ads_index, surf_index in atom_group_pairs:
            all_pairs.extend(product(ads_index, surf_index))
        self.all_pairs = all_pairs
        self.implemented_properties = ["energy", "forces"]
    
    def calculate(self,
                  atoms:Atoms=None,
                  properties=["energy", "forces"],
                  system_changes=all_changes):

        super().calculate(atoms)
        natoms = len(atoms)
        cell = atoms.get_cell()
        inv_cell = np.linalg.inv(cell)
        number = atoms.get_atomic_numbers()
        radius = covalent_radii[number]

        pos_ads = atoms.positions[self.nsurf_atoms:, :]     #吸附物原子的位置
        pos_remain = atoms.positions[0:self.nsurf_atoms, :] #表面原子的位置

        direct_ads = pos_ads @ inv_cell
        direct_remain = pos_remain @ inv_cell  
        diff_direct = direct_ads[:, np.newaxis, :] - direct_remain[np.newaxis, :, :]
        diff_direct = diff_direct - np.round(diff_direct)
        diff_cart = diff_direct @ cell 
        dist_cart = np.sqrt(np.sum(diff_cart**2, axis=2)) #(N_rigid, N_attra) 每个吸附物原子离表面原子的距离

        radius_ads = radius[self.nsurf_atoms:]
        radius_surf = radius[0:self.nsurf_atoms]
        radius_sum = radius_ads[:, np.newaxis] + radius_surf[np.newaxis, :]
        aa = 2 ** (1.0/6)

        flag = dist_cart > radius_sum * aa 

        d = dist_cart + 0.01 #(Nads, Nsurf)
        a = radius_sum / d   #(Nads, Nsurf)
        a6 = a**6            #(Nads, Nsurf)
        e_array = 4*a6*(a6-1) #(Nads, Nsurf)
        f_array = 24*a6*(2*a6-1)/d
        f_array = f_array[:,:,np.newaxis]*diff_cart #(Nads, Nsurf, 3)

        e_array[flag] = 0 # 把吸引区域的设为0，因为它们太小改成二次函数势
        f_array[flag] = 0 

        factor = 10.0
        E = np.sum(e_array)
        forces = np.zeros((natoms, 3))
        forces[self.nsurf_atoms:, :] = np.sum(f_array, axis=1)
        # 再遍历每个pair，如果大于最小值点的距离加二次函数吸引势
        for i, j in self.all_pairs:
            if dist_cart[i, j] > radius_sum[i, j] * aa:
                dr = dist_cart[i, j] - radius_sum[i, j] * aa
                E += factor*dr**2
                forces[i+self.nsurf_atoms, :] += -2*factor*dr*diff_cart[i, j]/dist_cart[i, j]
        
        forces[self.nsurf_atoms:, :] = rigid_body_forces(pos_ads, forces[self.nsurf_atoms:, :])

        self.results = {}
        self.results["energy"] = E
        self.results["forces"] = forces

 