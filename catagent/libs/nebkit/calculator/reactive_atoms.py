import numpy as np 
from typing import List

from ase import Atoms 
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes

class ReactiveAtoms(Calculator):
    '''
    After NEB interpolation, some atoms may be crowded together. When using this class, specify a subset of atoms as active atoms using their indices. These atoms will be subjected to the short-range repulsive part of the Lennard-Jones (LJ) potential for NEB calculations to obtain a reasonable NEB path structure.

    Attributes:
        atoms_index: The index of active atoms.
        scale: A scale factor to scale the equilibrium distance between two atoms. `scale = 1.0` will use the some of covalent radii.
    Examples:
        Suppose performing NEB calculation of CH4 dissociation and the atoms along NEB path are crowded together because of unreasonable interpolation. `CH4_index` is the index of CH4 atoms.
        >>> from ase.mep import NEB
        >>> from ase.optimize import MDMin
        >>> neb = NEB(images, k=5.0, method="improvedtangent") #Use a large spring constant
        >>> neb.interpolate(method="linear") # The atoms may be crowded together.
        >>> CH4_index = [0,1,2,3,4]
        >>> for image in images:
        ...     image.calc = ReactiveAtoms(CH4_index, 0.6)
        >>> opt = MDMin(neb)
        >>> opt.run(fmax=0.5, steps=100) 
    '''
    def __init__(self, atoms_index:List[int] | np.ndarray, scale=0.6, **kwargs):
        '''
        Args:
            atoms_index: The index of active atoms.
            scale: A scale factor to scale the equilibrium distance between two atoms. `scale = 1.0` will use the some of covalent radii. Recommended values are [0.5, 0.6].
        '''
        Calculator.__init__(self, **kwargs)
        self.atoms_index = atoms_index
        self.n_active_atoms = len(atoms_index)
        self.implemented_properties = ["energy", "forces"]
        self.scale = scale
    
    def calculate(self,
                  atoms:Atoms=None,
                  properties=["energy", "forces"],
                  system_changes=all_changes):

        super().calculate(atoms)
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_radius = covalent_radii[atomic_numbers]
        ii, jj, dd, DD = neighbor_list('ijdD', atoms, 2.0*np.max(atomic_radius) + 0.5)
        E = 0
        force_of_active_atoms = np.zeros((self.n_active_atoms, 3))
        for i in range(self.n_active_atoms):
            flag = ii == self.atoms_index[i]
            ra = atomic_radius[self.atoms_index[i]]
            target_atom_indices = jj[flag]
            dist = dd[flag]
            dist_vec = DD[flag]
            n_target_atoms = len(target_atom_indices)
            for j in range(n_target_atoms):
                rb = atomic_radius[target_atom_indices[j]]
                a6 = (self.scale*(ra + rb)/(dist[j] + 1.0e-3)) ** 6
                E += 4*a6*(a6)
                force_of_active_atoms[i] += -24*a6*(2*a6)/(dist[j] + 1.0e-3)*dist_vec[j]

        forces = np.zeros((len(atoms), 3))
        forces[self.atoms_index] = force_of_active_atoms
        self.results = {}
        self.results["energy"] = E
        self.results["forces"] = forces