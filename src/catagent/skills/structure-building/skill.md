---
name: structure-building
description: Use ase and pymatgen to build simple structures
metadata:
  tags:
    - ase
    - pymatgen
    - rdkit
---

# Build simple structures with ase and pymatgen 

Main rule of structure building: 
1. Atoms should not crowd together. `ase.data` provides a  `covalent_radii` dict containing the covalent radii of atoms. The sum of the covalent radii of two atoms is a good reference for the reasonable distance between atoms.
2. Molecules should not dissociate in vacuum, they only dissociate on a site of catalyst.
3. For NEB task, the atomic indices must be matched and orientation of fragments must be aligned. For example, for reaction "CH3 + CH2 <-> CH3CH2", if the indices are not matched or C-C atoms are not aligned, NEB interpolation will result in a very atom-crowded and twisted NEB path, NEB will never converge.
4. Except the mandatory requirements mentioned above, it's OK if the initial structure is not precisely physical. For example, it's not necessary to require CO2 adsorbs on surface with a bent configuration; it's also OK if the water layer you build dose not have a well-defined hydrogen bond network. As long as the atoms are not crowded together (satisfy 1,2 and 3), subsequent DFT or MLFF structure relaxation will guide to the structure to reasonable configuration. It's very usual to build a structure step by step. For example, when building a electrochemical interface, we firstly build a surface and put 10~20 H2O molecules above the surface. The orientations of these H2O molecules are not important, just randomly and evenly place them above the slab. Then we perform a quick relaxation by MLFF or cheap DFT (small ENCUT, loose EDIFF and EDIFFG, gamma only KPOINTS). 

NOTE: For structure building, also see the "nebkit" skill. Functions provided by nebkit can directly generate non-crowded IS and FS structures for NEB calculation, the atomic indices are matched, orientations of molecules are also aligned. They can be directly use in DFT calculation.

Below are basic usage of `ase` and `pymatgen` on structure building.
## Structure format conversion
You can use `ase` or `pymatgen` to convert a format into another, for example:
```bash
python3 << 'EOF'
from ase.io import read, write
# use `index` argument if you want a slice of the structures
atoms = read('OUTCAR', format='vasp-out', index=':') 
write('results.extxyz', atoms, format='extxyz')
EOF 
```
```bash
python3 << 'EOF'
from pymatgen.core import Structure
structure = Structure.from_file("POSCAR")
structure.to(filename="output.cif")
EOF
```

## Molecules
Use `scripts/build_molecule.py` to build a molecule by chemical formula, smiles string or molecular graph.
```bash
# By chemical formula
python3 build_molecule.py -o POSCAR formula CH3CHO

# By smiles string
python3 build_molecule.py -o C2H4.xyz smiles "C=C" --addH

# By molecular graph
python3 build_molecule.py -o POSCAR graph --vertexes 'H,C,O,O,H' --edges '0-1,1-2,1-3,3-4'
```
Use `-h` option to see the help messages.

## Bulk
Use `ase.build.bulk` to build simple bulk structures. For example:
```bash
python3 << 'EOF'
from ase.build import bulk
from ase.io import write
atoms = bulk(name='Pt', crystalstructure='fcc', cubic=True) # You can also set lattice constant here
write("POSCAR", atoms)
EOF
```
You can use `scripts/conv_prim.py` to convert between conventional cell and primitive cell.
```bash
# Conver conventional cell to primitive cell
python3 conv_prim.py --conv POSCAR -o POSCAR_prim

# Conver primitive cell to conventional cell
python3 conv_prim.py --prim structure.cif -o conv.cif
```
## Slab (surface)
Use `scripts/build_slab.py` to build a surface/slab.
```bash
# Build simple slab by ase
# Support fcc100, fcc110, fcc111, fcc211, bcc100, bcc110, bcc111, hcp0001 and hcp10m10
python3 build_slab.py -o POSCAR_Cu211 ase -e Cu -f fcc211 -s '3,3,6' -v 8 --nfix 3

# Build complex slab from a bulk structure by pymatgen
python3 build_slab.py -o POSCAR_result pymatgen --bulk POSCAR_NiO --miller '2,1,1' --size_ab 16.0 --size_c 8.0 --size_fix 4.0
```
`pymatgen` tends to build a tilted slab, for simple slabs of fcc, hcp and bcc, try `ase` first.

## Scripts
Below are the related scripts in `scripts` dir. Run these scripts without parameters or with `-h` to get the helps.

| Name | Purpose|
| ---- | ---- |
|conv_prim.py| Converting between conventional cell and primitive cell |
|build_molecule.py|Build molecules|
|build_slab.py|Build slabs (surface)|