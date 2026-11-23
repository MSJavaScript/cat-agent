
import numpy as np 
from typing import List, IO
from ase import Atoms 
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.optimize import Optimizer

from .tools import rigid_body_forces, adjust_positions


class RigidMDMin(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'dt': 0.2}

    def __init__(
        self,
        atoms: Atoms,
        index_group: List[np.ndarray],
        restart: str | None = None,
        logfile: IO | str = '-',
        trajectory: str | None = None,
        dt: float | None = None,
        maxstep: float | None = None,
        **kwargs,
    ):
        super().__init__(atoms, restart, logfile, trajectory, **kwargs)

        self.dt = dt or self.defaults['dt']
        self.maxstep = maxstep or self.defaults['maxstep']
        self.index_group = index_group

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, forces=None):
        forces = -self._get_gradient(forces)

        optimizable = self.optimizable

        if self.v is None:
            self.v = np.zeros(optimizable.ndofs())
        else:
            self.v += 0.5 * self.dt * forces
            # Correct velocities:
            vf = np.vdot(self.v, forces)
            if vf < 0.0:
                self.v[:] = 0.0
            else:
                self.v[:] = forces * vf / np.vdot(forces, forces)

        self.v += 0.5 * self.dt * forces
        pos = optimizable.get_x()
        dpos = self.dt * self.v
        maxstep = self.optimizable.gradient_norm(dpos)
        scaling = self.maxstep / (1e-6 + maxstep)
        dpos *= np.clip(scaling, 0.0, 1.0)
        
        pos = pos.reshape(-1, 3)
        dpos = dpos.reshape(-1, 3)
        new_pos = pos + dpos 
        for index in self.index_group:
            new_pos[index] = adjust_positions(pos[index], dpos[index])

        optimizable.set_x(new_pos.flatten())
        self.dump((self.v, self.dt))


class AlignIniFinAtoms(Calculator):
    '''
    这个类根据距离最小原则调整反应物和产物的取向。把反应物和产物分子都看成刚体，如果只有一个分子片则质心
    放在原点O，有两个分子片则放在原点的两侧。调用`calculate`方法时接受的Atoms对象是把反应物和产物都加到一起。
    这个类假设反应物分子在前，产物分子在后。设有n个原子，则反应物分子的索引是[0, natoms)，产物分子索引是[natoms, 2*natoms)
    如果有两个片段则它们的中心都在x轴上。
    '''
    def __init__(self, 
                 reactants_index:List[np.ndarray],  
                 products_index:List[np.ndarray], 
                 mapping_index:List[int],
                 combined_atoms:Atoms,
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        
        self.reactants_index = reactants_index
        self.products_index = products_index
        self.mapping_index = mapping_index
        self.natoms = len(mapping_index)
        self.implemented_properties = ["energy", "forces"]

        reactants_center = []
        for index in reactants_index:
            reactants_center.append(np.mean(combined_atoms.positions[index], axis=0))
        self.reactants_center = reactants_center
        self.reac_center_dist = None if len(reactants_center) == 1 else np.linalg.norm(reactants_center[0] - reactants_center[1])

        products_center = []
        for index in products_index:
            products_center.append(np.mean(combined_atoms.positions[index], axis=0))
        self.products_center = products_center
        self.prod_center_dist = None if len(products_center) == 1 else np.linalg.norm(products_center[0] - products_center[1])

    def calculate(self, 
                atoms:Atoms=None, 
                properties = ["energy", "forces"], 
                system_changes=all_changes):
        super().calculate(atoms)
        assert not np.any(atoms.pbc) #是纯分子，没有周期性边界条件

        natoms = self.natoms
        force_reac = np.zeros((natoms, 3))
        force_prod = np.zeros((natoms, 3))

        # 这里的原子序号需要匹配
        # 在对应的原子之间施加弹簧力使得原子距离尽可能小
        # 如果有两个分子片则需要防止两个分子片靠太近，所以还需要根据球心距离施加弹簧力
        E = 0
        for i in range(natoms):
            j = self.mapping_index[i]
            a, b = i, j + natoms
            vec = atoms.positions[a] - atoms.positions[b]
            force_reac[i] = -2*vec 
            force_prod[j] = 2*vec 
            E += np.linalg.norm(vec)

        k = 5
        if len(self.reactants_index) == 2:
            # 分别计算两个分子片所在的球，施加指向球心的力 
            center0 = np.mean(atoms.positions[self.reactants_index[0]], axis=0)
            center1 = np.mean(atoms.positions[self.reactants_index[1]], axis=0)
            dist = np.linalg.norm(center0 - center1)
            F = -2*k*(dist - self.reac_center_dist)*(center0 - center1)/dist
            force_reac[self.reactants_index[0]] += F/len(self.reactants_index[0])
            force_reac[self.reactants_index[1]] += -F/len(self.reactants_index[1])

        if len(self.products_index) == 2:
            # 分别计算两个分子片所在的球，施加指向球心的力 
            center0 = np.mean(atoms.positions[self.products_index[0]], axis=0)
            center1 = np.mean(atoms.positions[self.products_index[1]], axis=0)
            dist = np.linalg.norm(center0 - center1)
            F = -2*k*(dist - self.prod_center_dist)*(center0 - center1)/dist
            force_prod[self.products_index[0]-natoms] += F/len(self.products_index[0])
            force_prod[self.products_index[1]-natoms] += -F/len(self.products_index[1]) 
        
        start = 0
        for i in range(len(self.reactants_index)):
            n_sub_atoms = len(self.reactants_index[i])
            force_reac[start:(start+n_sub_atoms)] = rigid_body_forces(atoms.positions[self.reactants_index[i]], 
                                                                       force_reac[start:(start+n_sub_atoms)])
            start += n_sub_atoms
        
        start = 0
        for i in range(len(self.products_index)):
            n_sub_atoms = len(self.products_index[i])
            force_prod[start:(start+n_sub_atoms)] = rigid_body_forces(atoms.positions[self.products_index[i]],
                                                                       force_prod[start:(start+n_sub_atoms)])
            start += n_sub_atoms

        self.results = {}
        self.results["energy"] = E
        self.results["forces"] = np.vstack((force_reac, force_prod))



