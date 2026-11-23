---
name: nebkit
description: Usage of nebkit. Including structure-building and reaction network generation.
metadata:
  tags:
    - nebkit
    - structure-building
    - neb
    - transition state search 
    - reaction network
---

# Nebkit
Nebkit is a python package in development, it currently include the following functions:
- Given the reactants and products, enumerate the elementary reactions.
- Given a surface, enumerate adsorption configurations based on local environments of surface sites.
- NEB interpolation, automatically handle some special cases.


## Enumerate the elementary reactions

In nebkit, elementary reactions are generated according to the pure graph-theoretical method. In each step, only one chemical bond is changed, molecular fragments dissociate, combined, form a ring or a ring is broken, in this way, all the intermediates and elementary steps can be generated. The iteration stops when all the product molecules are generated and all intermediates are consumed by current elementary steps. This algorithm splits reactions such as atomic group transfer and insertion into multiple steps. For example: reactions NH2OH + H <-> NH2 + H2O will be (1) NH2OH <-> NH2 + OH, (2) H + OH <-> H2O, reaction CH3CH3 + CH2: <-> CH3CH2CH3 will be (1) CH3CH3 <-> CH3 + CH3 (2)CH3 + :CH2 <-> CH3CH2 (3)CH3CH2 + CH3 <-> CH3CH2CH3. To get a network consistent with chemical intuition, we need to merge such reactions and delete redundant ones. 

### Generate elementary steps
```python
# Generate reaction network 
from nebkit.network.system import ReactionSystem
from ase.build import molecule
H2 = molecule('H2')
N2 = molecule('N2')
NH3 = molecule('NH3')
# Each species only appear once, 
rs = ReactionSystem([H2, N2], [NH3])
rs.generate_elementary_steps(max_natoms_tot=4, max_natoms={'N': 1})
# If no reactions are generated, reactions_list will be empty.
for reac in rs.reactions_list:
    print(reac.reaction_str)
```
To limit the number of reactions in the network, you should set parameters of `generate_elementary_steps`:

```python
# definition of `generate_elementary_steps`
def generate_elementary_steps(self,
                           ntimes:int = -1,
                           max_natoms_tot:int = -1, 
                           min_natoms_tot:int = 0, 
                           max_natoms:Dict[str, int] = None, 
                           min_natoms:Dict[str, int] = None, 
                           forbidden_pattern: List[Tuple[str, str]] = None):
                           pass
```
- ntimes: recommend use default value `-1`, it will search all the possible steps (a BFS-like method). `ntimes > 0` means using random DFS search to generate reactions, this usually fails because we can only run DFS with finite steps.
- max_natoms_tot: Max totoal number of atoms in a intermediate.
- Min_natoms_tot: Min totoal number of atoms in a intermediate.
- max_natoms: A dict to set max number of atoms for each element.
- min_natoms: A dict to set min number of atoms for each element.
- forbidden_pattern: The forbidden substructure (see below).

```python
# use `forbidden_pattern` to set forbidden substructures.
H2 = molecule('H2')
CO2 = molecule("CO2")
CH4 = molecule("CH4")
CO = molecule("CO")
rs = ReactionSystem([CO2, CH4], [H2, CO])
rs.generate_elementary_steps(max_natoms={'C': 1}, 
    forbidden_pattern=[('O,O', '0-1'), ('H,O,C,O,H','0-1,1-2,2-3,3-4')])
for reac in rs.reactions_list:
     print(reac.reaction_str)

```
In the example, `('O,O', '0-1')` represents a peroxide substructure -O-O-, `('H,O,C,O,H','0-1,1-2,2-3,3-4')` represents a geminal diol substructure H-O-C-O-H. If `forbidden_pattern` is not set, default forbidden patterns are generated, see `nebkit.network.system._get_default_bad_patterns`.

### Merge reactions, delete reactions and intermediate species

You can merge reactions by using `rs.merge_reactions()`
```python
from ase.build import molecule 
from nebkit.network.system import ReactionSystem

H = molecule('H') # H represent a proton H^{+}
NO = molecule('NO')
H2O = molecule('H2O')
NH3 = molecule('NH3')
H2 = molecule('H2')
rs = ReactionSystem([H, NO], [H2O, NH3, H2])
rs.generate_elementary_steps(max_natoms={'N': 1, 'O':1})
for i, reac in enumerate(rs.reactions_list):
    print(i, reac.unique_reaction_str)
```
The output is 
```
0 [H] + [H] <-> [H][H]
1 [H] + NO <-> [H]NO
2 [H] + NO <-> [H]ON
3 NO <-> N + O
4 [H]NO + [H] <-> [H]N([H])O
5 [H]NO + [H] <-> [H]NO[H]
6 [H]ON + [H] <-> [H]NO[H]
7 N + [H] <-> [H]N
8 O + [H] <-> [H]O
9 [H]NO <-> [H]N + O
10 [H]ON <-> [H]O + N
11 [H]N + [H]O <-> [H]NO[H]
12 [H]NO[H] + [H] <-> [H]ON([H])[H]
13 [H]N + [H] <-> [H]N[H]
14 [H]O + [H] <-> [H]O[H]
15 [H]N([H])O + [H] <-> [H]ON([H])[H]
16 [H]N([H])O <-> [H]N[H] + O
17 [H]N[H] + [H] <-> [H]N([H])[H]
18 [H]N[H] + [H]O <-> [H]ON([H])[H]
```

For reactions 
```
10 [H]ON <-> [H]O + N
11 [H]N + [H]O <-> [H]NO[H]
18 [H]N[H] + [H]O <-> [H]ON([H])[H]
```
and reaction
```
14 [H]O + [HH] <-> [H]O[H]
```
There are usually merged, for example, `NOH <-> OH + N` and `OH + H <-> H2O` are merged to get `NOH + H <-> N + H2O`. To do this, use `merge_reactions`.

```python
rs.merge_reactions(10, 14)
rs.merge_reactions(11, 14)
rs.merge_reactions(18, 14)
for i, reac in enumerate(rs.reactions_list):
    print(i, reac.unique_reaction_str) 

rs.delete_reactions([11, 18, 10])
for i, reac in enumerate(rs.reactions_list):
    print(i, reac.unique_reaction_str) 
```
Because `NOH <-> OH + N` is a slow step, it is redundant with the total reaction `NOH + H <-> N + H2O`, we need to delete it. The deletion can not be done without reaction kinetic data, we left this to the user.

**NOTE: You can't merge arbitrary reactions. If the number of chemical bond changes between IS and FS are greater than 2, it can't be a reasonable elementary step.**

You can use `delete_reactions()` to delete reactions and use `delete_species()` to delete intermediate species. When some steps are deleted and some steps become isolated (as node in graph), these isolated steps will also be deleted. `delete_species()` also delete reactions which contain this species, if isolated steps appear, they will be deleted too.

**NOTE: Reaction and species indices will change after deletion. Make sure to only delete once or recalculate the indices after deletion.**

```python
H2 = molecule('H2')
CO2 = molecule("CO2")
CH4 = molecule("CH4")
CO = molecule("CO")
rs = ReactionSystem([CO2, CH4], [H2, CO])
rs.generate_elementary_steps(max_natoms={'C': 1, 'O':1})
for i, reac in enumerate(rs.reactions_list):
    print(i, reac.unique_reaction_str)

# Delete all the reactions contains O-O
rs.delete_reactions([30, 26, 24, 22, 21, 8, 0])

print(len(rs.reactions_list))
# Print degrees of unsaturation of each species
# Reactants CH4, CO2, Products CO, H2 and species with zero degree of unsaturation are allowed to appear in only one reaction.
print(rs.unsaturation_degree_list)
# Calculate the number of reactions in which a species appear. If an intermediate is unsaturated and only appears in one reaction, then the network is problematic. Because an intermediate must be produced and resumed in a connected network.
print(np.sum(rs.get_reactions_species_matrix(), axis=0))
```

### Render to pictures and dump import informations
Call `render_system()` to render the reactions and intermediate species into pngs.
```python
import os
os.makedirs("./visualize")
# recommend rdkit backend, more clear and concise
rs.render_system("./visualize", "rdkit")
```
Call `dump_info()` to dump import informations into a json file. This function also generates a `intermediates.extxyz` in the same location, containing all the the 3D structures of intermediates. The `reactant_indices` and `product_indices` fields in the json gives indices of reactant and product molecules in `intermediates.extxyz`.

### Build IS and FS on a surface
See next section.

## Structure building
Nebkit provides some functions for structure building, by now, only surface reactions are supported. These structures can be directly used in DFT or MLFF calculation (although the geometry is not precisely physical, e.g., CO2 is not bent on a surface, but this is OK. Because DFT or MLFF relaxation will correct it).

1. `.bind_to_surfaces()`: `ReactionSystem` has a `.bind_to_surfaces()` method, directly build IS and FS on the given surface, atomic indices are automatically matched and orientation of molecular fragments is also aligned. 

2. `bind_ini_fin_to_surface`:  Directly build IS and FS of a reaction, atomic indices are automatically matched and orientation of molecular fragments is also aligned. 

3. `bind_single_ads_to_surface`: Bind a single adsorbate to the surface.

4. `build_surface_neb_path`: Build NEB path. This function handle some special cases, see the doc string of this function.

## General workflow (Human in the loop)

You should call `pause_to_ask` (agent tool) to ask the user for key params of nebkit apis.

1. Before you generate elementary steps, call `pause_to_ask`  to ask for the constraints on the reaction network.

2. After network is generated. Call `render_system()` and `dump_info()` to render pictures, Then call `pause_to_ask` to let the user provide the indices of reactions or species they want to delete, or the reaction pairs they want to merge. The user may want to regenerate the network with different constraints, then you should goto 1.

3. Before you call `.bind_to_surfaces()` you should call `pause_to_ask` to ask the user if we are done with the network generation and editing. If they are confirmed, then call `.bind_to_surfaces()` to build structures.

4. There will be many configuration for complex surface (e.g. a step surface), call `pause_to_ask` to ask the user which surface they want in the calculation. NOTE: The IS and FS returned by `.bind_to_surfaces()` is corresponding to each other, you can safely use index to extract structures from the given list.