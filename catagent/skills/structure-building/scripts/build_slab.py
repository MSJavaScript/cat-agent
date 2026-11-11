"""
Build slab(surface) by ase or pymatgen 

Usage:
# Build simple slab by ase
python3 build_slab.py -o POSCAR_Cu211 ase -e Cu -f fcc211 -s '3,3,6' -v 8 --nfix 3

# Build complex slab from a bulk structure by pymatgen
python3 build_slab.py -o POSCAR_result pymatgen --bulk POSCAR_NiO --miller '2,1,1' --size_ab 16.0 --size_c 8.0 --size_fix 4.0
"""
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure

import ase
from ase import Atoms 
from ase.io import read, write
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from ase.build import fcc100, fcc110, fcc111, fcc211, bcc100, bcc110, bcc111, hcp0001, hcp10m10

from typing import Tuple, List
import numpy as np
import math
import argparse


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


def group_surface_layers(surface:Atoms, threshold:float = 0.25)->List[List[int]]: 
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
    #print(len(bins))
    bin_indices = np.digitize(z_n, bins, right=False)
    #print("bin_indices = ", np.max(bin_indices))
    result = []
    for i in range(1, len(bins)):
        indices = np.where(bin_indices == i)[0]
        if len(indices) != 0:
            result.append(indices)
    return result 


def from_bulk_get_specific_millers(
        specific_millers:Tuple[int, int, int], 
        bulk:Atoms, 
        min_ab=8.0, 
        min_slab_size:float = 7.0,
        min_vacuum_size:float = 20.0,
        in_unit_planes:bool = False
    ):
        assert isinstance(specific_millers, tuple)
        assert len(specific_millers) == 3
        assert bulk is not None

        untiled_slabs = compute_slabs(
            bulk,
            max_miller=max(np.abs(specific_millers)),
            specific_millers=[specific_millers],
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            in_unit_planes=in_unit_planes
        )
        slabs = [
            (
                tile_and_tag_atoms(s[0], bulk, min_ab=min_ab),
                s[1],
                s[2],
                s[3],
                s[4],
            )
            for s in untiled_slabs
        ]
 
        return slabs[0][0]


def tile_and_tag_atoms(
    unit_slab_struct: Structure, bulk_atoms: ase.Atoms, min_ab: float = 8
):
    slab_atoms = AseAtomsAdaptor.get_atoms(unit_slab_struct)
    return tile_atoms(slab_atoms, min_ab=min_ab)
    
def tile_atoms(atoms: ase.Atoms, min_ab: float = 8):
    a_length = np.linalg.norm(atoms.cell[0])
    b_length = np.linalg.norm(atoms.cell[1])
    na = int(math.ceil(min_ab / a_length))
    nb = int(math.ceil(min_ab / b_length))
    n_abc = (na, nb, 1)
    return atoms.repeat(n_abc)


def compute_slabs(
    bulk_atoms: ase.Atoms = None,
    max_miller: int = 2,
    specific_millers: list | None = None,
    min_slab_size:float = 7.0,
    min_vacuum_size:float = 20.0,
    in_unit_planes:bool = False):

    assert bulk_atoms is not None
    bulk_struct = standardize_bulk(bulk_atoms)

    if specific_millers is None:
        specific_millers = get_symmetrically_distinct_miller_indices(
            bulk_struct, max_miller
        )

    all_slabs_info = []
    for millers in specific_millers:
        slab_gen = SlabGenerator(
            initial_structure=bulk_struct,
            miller_index=millers,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=1,
            in_unit_planes=in_unit_planes
        )
        slabs = slab_gen.get_slabs(
            tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False
        )

        # If the bottom of the slabs are different than the tops, then we
        # want to consider them too.
        if len(slabs) != 0:
            flipped_slabs_info = [
                (flip_struct(slab), millers, slab.shift, False, slab.oriented_unit_cell)
                for slab in slabs
                if is_structure_invertible(slab) is False
            ]

            # Concatenate all the results together
            slabs_info = [
                (slab, millers, slab.shift, True, slab.oriented_unit_cell)
                for slab in slabs
            ]
            all_slabs_info.extend(slabs_info + flipped_slabs_info)

    return all_slabs_info


def flip_struct(struct: Structure):
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, "x", rotate_cell=True, center="COM")
    if atoms.cell[2][2] < 0.0:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.center()
    atoms.wrap()

    return AseAtomsAdaptor.get_structure(atoms)


def is_structure_invertible(struct: Structure):
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False


def standardize_bulk(atoms: ase.Atoms):
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    return sga.get_conventional_standard_structure()


parser = argparse.ArgumentParser(description="Build slab of specific Miller index.")
parser.add_argument("-o", "--outfile", type=str, required=True, help="The name of output file.")

subparsers = parser.add_subparsers(dest='package', help='The package used to build the slab.')

ase_parser = subparsers.add_parser("ase", help="Use ase to build simple slabs, such as fcc111, hcp0001, etc.")
ase_parser.add_argument("-e", "--element", type=str, required=True, help="A single chemical element.")
ase_parser.add_argument("-f", "--facet", type=str, required=True, help="The facet. Support fcc100, fcc110, fcc111, fcc211, bcc100, bcc110, bcc111, hcp0001 and hcp10m10.")
ase_parser.add_argument("-s", "--size", type=str, required=True, help="The size of slab, three integers separated by comma, for example, '6,4,5' is a 6*4, 5 layers slab.")
ase_parser.add_argument("-v", "--vacuum", type=float, default=8.0, help="The vacuum size. Vacuum region is along c-direction and slab is centered. Default 8.0.")
ase_parser.add_argument("--nfix", type=int, help="The number of layers you want to fix the positions.")
ase_parser.add_argument("-a", "--a_length", type=float, help="Lattice constant in `a` direction.")
ase_parser.add_argument("-c", "--c_length", type=float, help="Lattice constant in `c` direction.")


pymatgen_parser = subparsers.add_parser("pymatgen", help="Use pymatgen to build complex slab from a bulk structure.")
pymatgen_parser.add_argument("--bulk", required=True, type=str, help="Path to bulk structure.")
pymatgen_parser.add_argument("--miller", required=True, type=str, help="Miller index, three integers separated by comma. For example, '2,1,1', '1,1,1', etc.")

pymatgen_parser.add_argument("--size_ab", type=float, default=8.0, help="Length in ab direction (Angstroms). This value should be larger than the diameter of adsorbates to avoid overlapping of atoms in periodic images. Default 8.0.")
pymatgen_parser.add_argument("--size_c", type=float, default=8.0, help="The thickness of slab in c direction (Angstroms). Default 8.0.")
pymatgen_parser.add_argument("--size_fix", type=float, default=3.5, help="Thickness of slab requiring to fix atomic positions. Default 3.5.")
pymatgen_parser.add_argument("--natoms", type=int, help="The limitation of the number of atoms in the slab. ")


args = parser.parse_args()

if args.package == 'ase':
    element = args.element
    facet = args.facet
    size = args.size
    vacuum = args.vacuum
    a_length = args.a_length
    c_length = args.c_length
    nfix = args.nfix

    supported_facets = {"fcc100":fcc100, "fcc110":fcc110, "fcc111":fcc111, "fcc211":fcc211, 
                        "bcc100":bcc100, "bcc110":bcc110, "bcc111":bcc111, "hcp0001":hcp0001, "hcp10m10":hcp10m10}

    size = size.replace(' ', '').split(',')
    size = (int(size[0]), int(size[1]), int(size[2]))
    
    if not facet in supported_facets:
        print(f"Error: facet {facet} is not supported. Only simple facets {list(supported_facets.keys())} are supported.")
        exit(0)
        
    try:
        if 'hcp' in facet:
            surface:Atoms = supported_facets[facet](symbol=element, 
                                              size=(size[0], size[1], size[2]), 
                                              vacuum=vacuum,
                                              a=a_length, c=c_length)
        else:
            surface:Atoms = supported_facets[facet](symbol=element, 
                                              size=(size[0], size[1], size[2]), 
                                              vacuum=vacuum,
                                              a=a_length)
    except Exception as e:
        print(f"Error: Cannot build {facet} of {element}. \n {e}")
        exit(0)
    
    if nfix:
        if nfix >= size[2]:
            print(f"Error: The number of fixed layers {nfix} must be less than the total number of layers {size[2]}.")
            exit(0)
        tags = surface.get_tags()
        if np.all(tags[0] == tags): #所有的tag都相同无法分层
            groups = group_surface_layers(surface, 0.25)
            nfix_group = int(float(len(groups)) * nfix / size[2])
            fix_indices = np.concatenate(groups[0:nfix_group])
            surface.set_constraint(FixAtoms(indices=fix_indices))
        else:
            unique_tags = np.unique(tags)[::-1]
            tags_to_fix = unique_tags[0:nfix]
            surface.set_constraint(FixAtoms(indices=[i for i, tag in enumerate(tags) if tag in tags_to_fix]))

    write(args.outfile, surface, format="vasp")
    print(f"{args.outfile} is generated.")
else:
    miller = args.miller
    size_ab = args.size_ab
    size_c = args.size_c
    fix_slab_size = args.size_fix
    max_natoms = args.natoms if args.natoms else -1

    miller = miller.replace(' ', '').split(',')
    miller = (int(miller[0]), int(miller[1]), int(miller[2]))

    atoms = from_bulk_get_specific_millers(specific_millers=miller,
                                           bulk=read(args.bulk),
                                           min_ab=size_ab,
                                           min_slab_size=size_c,
                                           min_vacuum_size=16,
                                           in_unit_planes=False)


    tmp_atoms = atoms.copy()
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
    n_fix_layer = round(float(fix_slab_size) / float(size_c)*n_remain_layer)

    collect_index = []
    collect_tag = []

    for i in range(i_group, ngroup):
        collect_index.extend(groups[i].tolist())
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
    formula = new_atoms.get_chemical_formula()
    write(args.outfile, new_atoms, format="vasp")
    print(f"{args.outfile} is generated.")