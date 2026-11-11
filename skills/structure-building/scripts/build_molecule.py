"""
Build molecule by chemical formula, smiles string or molecular graph. Can be saved in xyz or POSCAR format.

Usage:
# By chemical formula
python3 build_molecule.py -o POSCAR formula CH3CHO

# By smiles string
python3 build_molecule.py -o C2H4.xyz smiles "C=C" --addH

# By molecular graph
python3 build_molecule.py -o POSCAR graph --vertexes 'H,C,O,O,H' --edges '0-1,1-2,1-3,3-4'
"""
import argparse
from ase.build import molecule
from ase.collections import g2
from ase.io import write 
from ase import Atoms 
from ase.data import covalent_radii

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
import numpy as np 

def calculate_diameter_of_molecule(atoms):
    max_radius = np.max([ covalent_radii[a] for a in atoms.get_atomic_numbers() ])
    dist_mat = atoms.get_all_distances(mic=True)
    max_dist = np.max(dist_mat) + max_radius*2
    return max_dist


parser = argparse.ArgumentParser(description="Build simple molecules by molecule formula, smiles string or graph.")
parser.add_argument("-o", "--outfile", type=str, required=True, help="Output file name. Can be outputed in xyz or POSCAR format.")

subparsers = parser.add_subparsers(dest='type', help='Build type, use chemical formula, smiles or molecular graph.')

formula_parser = subparsers.add_parser("formula", help="Build with chemical formula.")
formula_parser.add_argument("formula", type=str, help="Chemical formula of the molecule. Use a single element name to create a single atom.")

smiles_parser = subparsers.add_parser("smiles", help="Build with smiles string.")
smiles_parser.add_argument("smiles", type=str, help="SMILES string.")
smiles_parser.add_argument("--addH", action="store_true", help="Add H atoms automatically, only valid when --smiles option is used.")

graph_parser = subparsers.add_parser("graph", help="Build with chemical molecular graph.")
graph_parser.add_argument("--vertexes", type=str, help="A list of chemical elements separated by comma. For example, 'H,C,O,O,H'. ")
graph_parser.add_argument("--edges", type=str, help="A list of edges in the following format: '0-1,1-2,1-3,3-4', the integers are the indices of chemical symbols set by --vertexes option.")
graph_parser.add_argument("--addH", action="store_true", help="Add H atoms automatically, only valid when --smiles option is used.")

args = parser.parse_args()

if (not args.outfile.endswith('.xyz') ) and (not args.outfile.startswith('POSCAR')):
    print(f"Error: Output file must be 'xyz' or POSCAR format. Got {args.outfile}.")
    exit(0)

if args.type == 'formula':
    formula = args.formula 
    if not formula in g2.names:
        print(f"Error: Molecule {formula} is not included in ASE g2 collection. ")
        exit(0)
    mol_atoms = molecule(formula)
elif args.type == 'smiles':
    smiles = args.smiles
    addH = args.addH
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Error: Cannot create molecule from SMILES {smiles}. Please check the SMILES.")
        exit(0)
    
    if addH:
        mol = Chem.AddHs(mol)
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        AllChem.EmbedMolecule(mol, params)
    mol_atoms = Atoms(numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
          positions = mol.GetConformer().GetPositions())
else:
    symbols = args.vertexes.replace(' ', '')
    edges = args.edges.replace(' ', '')
    addH = args.addH

    symbols_list = symbols.split(",")
    edges_list = edges.split(",")

    mol = Chem.RWMol()
    for s in symbols_list:
        mol.AddAtom(Chem.Atom(s))
    for edge in edges_list:
        a,b = edge.split('-')
        mol.AddBond(int(a), int(b), Chem.BondType.SINGLE)

    if addH:
        mol = Chem.AddHs(mol)
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        AllChem.EmbedMolecule(mol, params)
    mol_atoms = Atoms(numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
          positions = mol.GetConformer().GetPositions())
    

if 'xyz' in args.outfile:
    write(args.outfile, mol_atoms, format="xyz")
    print(f"{args.outfile} is generated.")
elif 'POSCAR' in args.outfile:
    d = calculate_diameter_of_molecule(mol_atoms)
    L = d * 2.5 
    L = 12.0 if L < 12.0 else L
    mol_atoms.set_cell([[L, 0.0, 0.0], [0.0, L+0.5, 0.0], [0.0, 0.0, L+1.0]])
    mol_atoms.center()
    write(args.outfile, mol_atoms, format="vasp")
    print(f"{args.outfile} is generated.")
else:
    pass 