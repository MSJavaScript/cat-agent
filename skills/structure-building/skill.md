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

Building structure is a difficult task. `ase` and `pymatgen` only provide some simple and standard functions for structure building, for complex cases, you should use specialized tools or ask the user to provide structures. This document demonstrates examples of building simple structures using `ase` and `pymatgen`, and provides some simple wrapper scripts in the `scripts` dir for calling.

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