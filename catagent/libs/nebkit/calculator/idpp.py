
from ase.mep import NEB 
from ase.optimize import MDMin
from ase.geometry import find_mic
from ase.calculators.calculator import Calculator
import numpy as np 

# 假设初态是平躺着的C-O（设键长为2.0），末态是旋转了180度的O-C（
# 即使它们移动了位置）直接线性插值的时候中间态的image C和O会挤在一起。
# 而idpp方法首先计算初态和末态之间每对儿原子间的距离，然后对距离进行插值。
# 从而中间态image中C-O的距离也是2.0，这个就是target的距离。在IDPP calculator
# 中，计算当前状态下每对儿原子的距离，对于线性插值的C-O假设是0。则此时
# C-O 之间有排斥势
# V = 0.5 * \sum_{(i,j)} (d_{ij} - d_{ij, target})^2 / d_{ij}^4
# 其中 (i,j) 表示原子对儿。分母上除以d_{ij}是为了防止原子距离过近。
# 这里的 idpp_interpolate 函数几乎照搬了ase的函数，但加了adsorbate_atom_index
# 参数。设置该参数可以让其他原子受力为0，从而避免表面原子在插值过程中移动。

def idpp_interpolate(images, adsorbate_atom_index=None, mic=False):
    neb = NEB(images, method="improvedtangent")
    d1 = neb.images[0].get_all_distances(mic=mic)
    d2 = neb.images[-1].get_all_distances(mic=mic)
    d = (d2 - d1) / (neb.nimages - 1)
    real_calcs = []
    for i, image in enumerate(neb.images):
        real_calcs.append(image.calc)
        image.calc = IDPP(d1 + i * d, mic=mic, adsorbate_atom_index=adsorbate_atom_index)
    
    opt = MDMin(neb, logfile="idpp.log", trajectory="idpp.traj")
    opt.run(fmax=0.2, steps=100)

    return neb.images

class IDPP(Calculator):
    """Image dependent pair potential.

    See:
        Improved initial guess for minimum energy path calculations.
        Søren Smidstrup, Andreas Pedersen, Kurt Stokbro and Hannes Jónsson
        Chem. Phys. 140, 214106 (2014)
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, target, mic, adsorbate_atom_index):
        Calculator.__init__(self)
        self.target = target
        self.mic = mic
        self.adsorbate_atom_index = adsorbate_atom_index

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        P = atoms.get_positions()
        d = []
        D = []
        for p in P:
            Di = P - p
            if self.mic:
                Di, di = find_mic(Di, atoms.get_cell(), atoms.get_pbc())
            else:
                di = np.sqrt((Di ** 2).sum(1))
            d.append(di)
            D.append(Di)
        d = np.array(d)
        D = np.array(D)

        dd = d - self.target
        d.ravel()[::len(d) + 1] = 1  # avoid dividing by zero
        d4 = d ** 4
        e = 0.5 * (dd ** 2 / d4).sum()
        f = -2 * ((dd * (1 - 2 * dd / d) / d ** 5)[..., np.newaxis] * D).sum(
            0)
        
        if self.adsorbate_atom_index:
            # 如果设置了吸附物原子索引，把除了这些原子之外其他的设置为0
            natoms = len(atoms)
            surface_atoms_index = np.setdiff1d(np.arange(natoms), self.adsorbate_atom_index)
            f[surface_atoms_index] = 0.0
        
        self.results = {'energy': e, 'forces': f}
