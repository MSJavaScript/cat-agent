from __future__ import annotations
from ase import Atoms 
from ase.io import write 
from ase.data import covalent_radii
from rdkit import Chem
import numpy as np
from typing import List, Tuple 
from catagent.libs.nebkit.tools.utils import mol_to_ase_atoms
from catagent.libs.nebkit.calculator.match_reactants_and_products import RigidMDMin, AlignIniFinAtoms

class ReactionHelper:
    def __init__(self, 
                 reactants_mols:List[Chem.Mol],
                 products_mols: List[Chem.Mol],
                 mapping_atom_index:np.ndarray,
                 debug=False):
        
        # 反应物和产物数目不超过2个，因为3分子反应几乎不可能发生。
        self.debug = debug 
        if not debug:
            assert len(reactants_mols) <= 2 and len(products_mols) <= 2
        
        self.reactants_mols   = reactants_mols
        self.products_mols    = products_mols
        self.reactants_atoms  = [mol_to_ase_atoms(mol) for mol in reactants_mols]
        self.products_atoms   = [mol_to_ase_atoms(mol) for mol in products_mols]
        self.reactants_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in reactants_mols]
        self.products_smiles  = [Chem.MolToSmiles(mol, canonical=True) for mol in products_mols]
        self.mapping_atom_index = mapping_atom_index

        # 把产物按照顺序排列
        sort_index = np.lexsort((mapping_atom_index[2], mapping_atom_index[1]))
        reverse_mapping_index = np.zeros_like(mapping_atom_index)
        natoms = mapping_atom_index.shape[1]
        i_mol = -1
        atom_in_reactant_mol_index = []
        for i in range(natoms):
            if mapping_atom_index[0, i] == 0:
                i_mol += 1
            atom_in_reactant_mol_index.append(i_mol)
        for i in range(natoms):
            j = sort_index[i]
            reverse_mapping_index[0, i] = mapping_atom_index[2, j]
            reverse_mapping_index[1, i] = atom_in_reactant_mol_index[j]
            reverse_mapping_index[2, i] = mapping_atom_index[0, j]
        self.reverse_mapping_index = reverse_mapping_index


        self.single_bond_reaction = True 

        a = [atoms.get_chemical_formula() for atoms in self.reactants_atoms]
        b = [atoms.get_chemical_formula() for atoms in self.products_atoms]
        self.reaction_str = ' + '.join(a) + ' <-> ' + ' + '.join(b)
        self.unique_reaction_str = ' + '.join(self.reactants_smiles) + ' <-> ' + ' + '.join(self.products_smiles)

        self.fragments_info = None 


    def mergeReaction(self, reaction:ReactionHelper) -> ReactionHelper:
        '''
        合并两个基元反应。有以下限制：
        1. self和reaction的反应物和产物分子数都小于等于2。
        2. reaction中只能有一个化学键变化。
        3. self和reaction有且仅有一个公共物种。
        ''' 
        def addReaction(forward:bool):
            if (forward): #尝试正向相加
                total_reactants_smiles = self.reactants_smiles + reaction.reactants_smiles
                total_products_smiles = self.products_smiles + reaction.products_smiles
            else:
                total_reactants_smiles = self.reactants_smiles + reaction.products_smiles
                total_products_smiles = self.products_smiles + reaction.reactants_smiles

            n_total_reac = len(total_reactants_smiles)
            n_total_prod = len(total_products_smiles)
            common_species:List[Tuple[int, int]] = []
            j_set = set()
            for i in range(n_total_reac):
                for j in range(n_total_prod):
                    if total_reactants_smiles[i] == total_products_smiles[j] and (not j in j_set):
                        common_species.append((i,j))
                        j_set.add(j)
                        break 

            if len(common_species) > 1:
                return None 
            
            # 只有一个公共物种且消去该物种之后等式两边分子数都不超过2
            if len(common_species) == 1:
                if not self.debug and (n_total_reac - 1 > 2 or n_total_prod - 1 > 2):
                    return None 

                if (forward):
                    total_reactants_mols = self.reactants_mols + reaction.reactants_mols
                    total_products_mols  = self.products_mols + reaction.products_mols
                else:
                    total_reactants_mols = self.reactants_mols + reaction.products_mols
                    total_products_mols  = self.products_mols + reaction.reactants_mols

                total_reactants_natoms = np.array([mol.GetNumAtoms() for mol in total_reactants_mols], dtype=int)
                cum_sum_total_reactants_natoms = np.hstack((0, np.cumsum(total_reactants_natoms)))

                self_n_products = len(self.products_smiles)
                other_mapping_atom_index = reaction.mapping_atom_index.copy() if forward else reaction.reverse_mapping_index.copy()
                other_mapping_atom_index[1, :] += self_n_products # 把reaction对应的产物编号加上偏移

                new_mapping_atom_index = np.hstack((self.mapping_atom_index, other_mapping_atom_index))

                index_of_common_in_tot_reac, index_of_common_in_tot_prod = common_species[0]
                # 进行原子匹配：产物中的原子在反应物中的位置
                mapping_array = np.array(total_reactants_mols[index_of_common_in_tot_reac].GetSubstructMatch(
                    total_products_mols[index_of_common_in_tot_prod]))
                
                # 获取产物分子的原子在new_mapping_atom_index中的位置
                atoms_pos_of_common_for_prod = np.where(new_mapping_atom_index[1, :] == index_of_common_in_tot_prod)[0]
 
                # 根据mapping_array得到映射之后产物原子的编号
                atoms_index_of_common_prod_after_mapping = [
                    mapping_array[new_mapping_atom_index[2, a]] for a in atoms_pos_of_common_for_prod
                ]

                # 再得到common物种在反应物处的原子编号，这个编号是连续的
                ss = slice(cum_sum_total_reactants_natoms[index_of_common_in_tot_reac], 
                        cum_sum_total_reactants_natoms[index_of_common_in_tot_reac+1])
                atoms_index_of_common_reac = new_mapping_atom_index[0, ss]
                # 根据 atoms_index_of_common_prod_after_mapping 调整 atoms_index_of_common_reac 的顺序
                re_ordered_index = np.array([ np.where(atoms_index_of_common_prod_after_mapping == a)[0][0] for a in atoms_index_of_common_reac])

                new_mapping_atom_index[1, atoms_pos_of_common_for_prod] = new_mapping_atom_index[1, ss][re_ordered_index]
                new_mapping_atom_index[2, atoms_pos_of_common_for_prod] = new_mapping_atom_index[2, ss][re_ordered_index]

                new_mapping_atom_index[:, ss] = -1
                mask = ~np.all(new_mapping_atom_index == -1, axis=0)
                new_mapping_atom_index = new_mapping_atom_index[:, mask]

                max_id = np.max(new_mapping_atom_index[1, :])
                for i in range(max_id):
                    if not np.any(new_mapping_atom_index[1, :] == i):
                        for j in range(new_mapping_atom_index.shape[1]):
                            if new_mapping_atom_index[1, j] > i:
                                new_mapping_atom_index[1, j] -= 1

                new_reactants_mols = [total_reactants_mols[i] for i in range(n_total_reac) if i != index_of_common_in_tot_reac]
                new_products_mols  = [total_products_mols[i] for i in range(n_total_prod) if i != index_of_common_in_tot_prod]
                rnc = ReactionHelper(reactants_mols=new_reactants_mols,
                                            products_mols=new_products_mols,
                                            mapping_atom_index=new_mapping_atom_index,
                                            debug=self.debug) 
                rnc.single_bond_reaction = False 
                return rnc 
            return None 
        
        if not reaction.single_bond_reaction: # 作为参数的反应只能变一根键
            raise ValueError("The number of bond change in a reaction must be 1.")
        result = addReaction(True)
        if not result is None:
            return result
        result = addReaction(False)
        if not result is None:
            return result
        return None 


   
    def get_aligned_reactants_and_products(self) -> Tuple[List[int], Atoms, Atoms]:
        '''
        基元反应发生需要满足一定的构象，特别是如果初态或末态有两个分子片。
        该函数把反应物和产物当作刚体让它们绕质心旋转，使得初末态距离最小从而进行对齐。
        '''  
        # 这里根据情况调整反应物或产物的重心      
        reactants_combined = _combine_atoms(self.reactants_atoms)
        products_combined  = _combine_atoms(self.products_atoms)

        nreac = len(self.reactants_atoms)
        nprod = len(self.products_atoms)

        if nreac == 1 and nprod == 2:
            reactants_combined.positions +=  products_combined.get_center_of_mass()
        elif nreac == 2 and nprod == 1:
            products_combined.positions += reactants_combined.get_center_of_mass()

        start = 0
        reactants_index = []
        products_index = []
        for atoms in self.reactants_atoms:
            reactants_index.append(start + np.arange(len(atoms)))
            start += len(atoms)
        for atoms in self.products_atoms:
            products_index.append(start + np.arange(len(atoms)))
            start += len(atoms)

        total_atoms = reactants_combined + products_combined
         # 确定原子对应的索引
        natoms = len(reactants_combined)
        n = 0
        prod_index_dict = dict()
        for i, atoms in enumerate(self.products_atoms):
            for j in range(len(atoms)):
                prod_index_dict[(i, j)] = n 
                n += 1
        mapping_index = [
            prod_index_dict[(int(self.mapping_atom_index[1, i]), int(self.mapping_atom_index[2, i]))]
             for i in range(natoms) ]
        
        total_atoms.calc = AlignIniFinAtoms(reactants_index, products_index, mapping_index, total_atoms)
        opt = RigidMDMin(total_atoms, 
                         index_group=reactants_index + products_index,
                         logfile="RigidMDMin.log", 
                         trajectory="RigidMDMin.traj")
        opt.run(fmax=0.2, steps=50)
        reactants_combined.set_positions(total_atoms.positions[0:natoms])
        products_combined.set_positions(total_atoms.positions[natoms:(2*natoms)])
        return mapping_index, reactants_combined, products_combined

        

    def __repr__(self):
        return self.reaction_str
    
    def __str__(self):
        return self.unique_reaction_str
    
    def _dump(self, filename_prefix:str):
        write(f"{filename_prefix}_reactants.extxyz", self.reactants_atoms, format="extxyz")
        write(f"{filename_prefix}_products.extxyz", self.products_atoms, format="extxyz")
        ff = open(f"{filename_prefix}_index.txt", mode="w")
        a = [str(i) for i in self.mapping_atom_index[0].tolist()]
        ff.write(','.join(a)+'\n')
        a = [str(i) for i in self.mapping_atom_index[1].tolist()]
        ff.write(','.join(a)+'\n')
        ff.close()



def _combine_atoms(list_of_atoms:List[Atoms]):
    if len(list_of_atoms) == 1:
        atoms = list_of_atoms[0].copy()
        center = np.mean(atoms.positions, axis=0)
        atoms.positions -= center 
        return atoms
    if len(list_of_atoms) == 2:
        # 首先确定每个分子片的直径
        atoms1 = list_of_atoms[0].copy()
        atoms1.rotate(np.random.random()*90-45, np.random.random((3, ))-0.5)
        max_radius1 = np.max([covalent_radii[a] for a in atoms1.get_atomic_numbers()])
        diameter1 = np.max(atoms1.get_all_distances()) + 2.0*max_radius1

        atoms2 = list_of_atoms[1].copy()
        atoms2.rotate(np.random.random()*90-45, np.random.random((3, ))-0.5)
        max_radius2 = np.max([covalent_radii[a] for a in atoms2.get_atomic_numbers()])
        diameter2 = np.max(atoms2.get_all_distances()) + 2.0*max_radius2

        dist = max(4.0, 0.5*(diameter1 + diameter2) + max(max_radius1, max_radius2))

        center1 = np.array([-0.5*dist, 0, 0])
        center = np.mean(atoms1.positions, axis=0)
        atoms1.positions -= center
        atoms1.positions += center1 
        
        center2 = np.array([0.5*dist, 0, 0])
        center = np.mean(atoms2.positions, axis=0)
        atoms2.positions -= center
        atoms2.positions += center2

        atoms1.extend(atoms2)
        return atoms1
    raise ValueError("Only no more than two fragments are allowed.")

