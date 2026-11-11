from typing import Tuple
from ase import Atoms 
from ase.constraints import FixAtoms

from nebkit.tools.structure import sort_elements, group_surface_layers
from fairchem.slab import Slab 
from fairchem.bulk import Bulk

def build_surface_from_bulk(
        bulk_atoms: Atoms,
        miller_indices: Tuple[int, int, int],
        min_ab: float = 8.0,
        min_slab_size: float = 8.0,
        fix_slab_size: float = 3.5,
        max_natoms: int = -1) -> Atoms:
    '''
    Construct complex surfaces based on bulk structures, such as alloy, oxide surfaces, or surfaces with high Miller indices. Total thickness of slab (`min_slab_size`), thickness of slab requiring fixed atomic positions (`fix_slab_size`) and size in ab directions (`min_ab`) are all determined by length (in Angstroms). 

    Args:
        bulk_atoms: `ase.Atoms` object of bulk.
        miller_indices: A tuple of int with length 3.
        min_ab: Min length in ab direction in Angstroms. `min_ab` should be larger than the diameter of adsorbates to avoid periodic image interaction.
        min_slab_size: Min thickness of slab in Angstroms.
        fix_slab_size: Thickness of slab requiring to fix atomic positions.
        max_natoms: Control the number of atoms in the slab. If `max_natoms > 0`, delete some atoms layer by layer from below.
    ''' 
    slabs = Slab.from_bulk_get_specific_millers(specific_millers=miller_indices, 
                                                bulk=Bulk(bulk_atoms=bulk_atoms), 
                                                min_ab=min_ab, 
                                                min_slab_size=min_slab_size, 
                                                in_unit_planes=False)

    slab = slabs[0]
    tmp_atoms = slab.atoms.copy()
    natoms = len(tmp_atoms)
    tmp_atoms.constraints = []  # remove all the constraints
    tmp_atoms = sort_elements(tmp_atoms)
    groups = group_surface_layers(tmp_atoms, 0.25)
    ngroup = len(groups)

    if max_natoms > 0 and natoms > max_natoms: #delete some atoms
        count = 0
        for i in range(ngroup-1, -1, -1):
            count += len(groups[i])
            if count > max_natoms:
                break 
        i_group = i+1 if (ngroup-i) > 2 else i
    else:
        i_group = 0
    n_remain_layer = ngroup - i_group
    n_fix_layer = round(float(fix_slab_size) / float(min_slab_size)*n_remain_layer)

    collect_index = []
    collect_tag = []

    for i in range(i_group, ngroup):
        collect_index.extend(groups[i])
        if i+1 - i_group <= n_fix_layer:
            collect_tag.extend([0]*len(groups[i]))
        else:
            collect_tag.extend([1]*len(groups[i]))
    
    new_atoms = Atoms(symbols=tmp_atoms.symbols[collect_index],
                    positions=tmp_atoms.positions[collect_index],
                    cell = tmp_atoms.cell,
                    pbc = [True, True, True],
                    tags=collect_tag)
    new_atoms.set_constraint(FixAtoms([i for i in range(len(new_atoms)) if new_atoms[i].tag == 0]))
    return new_atoms


 