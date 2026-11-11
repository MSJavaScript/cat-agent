from ase import Atoms 
from ase.formula import Formula
from ase.io import write 
from ase.data import covalent_radii

from typing import List, Tuple, Dict, Set 

from rdkit import Chem 
import numpy as np
import json, os

from .generate import ExploreReactionNetwork
from .helper import ReactionHelper
from libs.nebkit.tools.utils import ase_atoms_to_mol, get_adsorbate_binding_atom_index
from libs.nebkit.structure.bind_to_surface import get_outmost_atoms_for_surface, get_gcn, _bind_ini_fin_to_surface
from libs.nebkit.tools.plot_reaction import visualize_reaction, visualize_reaction_rdkit, draw_molecules_array



def _get_default_bad_patterns(elements:List[str], max_natoms:Dict[str, int]) -> List[Tuple[str, str]]:
    '''
    A helper function to generate the forbidden patterns.
    '''
    has_N = 'N' in elements
    has_O = 'O' in elements
    has_C = 'C' in elements

    max_N = max_natoms['N'] if 'N' in max_natoms else 100
    max_O = max_natoms['O'] if 'O' in max_natoms else 100 
    bad_pattern = []

    if has_N and max_N >= 2:
        bad_pattern.append(('N,N,N','0-1,1-2'))
        if has_O:
            bad_pattern.append(('O,N,N', '0-1,1-2'))
            bad_pattern.append(('N,O,N', '0-1,1-2'))
            bad_pattern.append(('O,N,N', '0-1,0-2,1-2'))
        if has_C:
            bad_pattern.append(('C,N,N', '0-1,0-2,1-2'))
            bad_pattern.append(('N,C,N,N', '0-1,1-2,2-3'))
    
    if has_O and max_O >= 2:
        bad_pattern.append(('O,O,O','0-1,1-2'))
        if has_N:
            bad_pattern.append(('N,O,O', '0-1,1-2'))
            bad_pattern.append(('O,N,O', '0-1,1-2'))
            bad_pattern.append(('N,O,O', '0-1,0-2,1-2'))
        if has_C:
            bad_pattern.append(('C,O,O', '0-1,0-2,1-2'))
            bad_pattern.append(('O,C,O,O', '0-1,1-2,2-3'))
    return bad_pattern

def _get_default_max_natoms(products:List[Atoms]):
    ele_dict = dict()
    for product in products:
        count = Formula(product.get_chemical_formula()).count()
        for key in count.keys():
            if key == 'H':
                continue 
            if not key in ele_dict:
                ele_dict[key] = count[key]
            elif ele_dict[key] < count[key]:
                ele_dict[key] = count[key]
    return ele_dict

class ReactionSystem:
    def __init__(self, reactants:List[Atoms],  products:List[Atoms]):
        self.reactants = reactants
        self.products = products 
        # 获取所有的元素
        elements = set()
        for atoms in reactants:
            elements.update(atoms.get_chemical_symbols())
        self.elements = sorted(elements)

        self.reactants_products_indices = []

        # 物种的atoms对象。这个成员用于批量构建中间物种在表面上的吸附态，
        # 以便于使用热力学方法进行反应路径的初筛
        self.species_atoms_list = []
        self.species_mols_list = []

        # 每个物种的smiles表示
        self.species_smiles_list = []

        # 用索引表示吸附物，一个tuple表示一个反应。如果一个反应对应的tuple中反应物
        # 或产物只有一个，那么在进行NEB计算时该物种可以避免计算，只去构建共吸附态
        # 注意这里有一个原子编号匹配的问题。
        # 这个数据成员也用于吸附物和反应的删除操作
        self.reaction_num_tuple = [] 

        # 其中每个元素是一个reaction helper。ReactionHelper 的职责是进行反应的合并
        # 以及构建反应的初末态。虽然 self.reaction_num_tuple 存放了每个反应的反应物
        # 和产物在self.species_atoms_list 中的索引。但注意self.species_atoms_list 是
        # 不能用于直接构建初末态的。因为同一个物种生成方式不同时其中的原子编号就不同，
        # 需要重新匹配原子序号。
        self.reactions_list:List[ReactionHelper] = [] 
        

    def dump_info(self, file_name:str):
        assert file_name.endswith('json')
        reactions = []
        for i, reac in enumerate(self.reactions_list):
            reactions.append({
                "id": i,
                "reac_str": reac.reaction_str,
                "reactant_indices":self.reaction_num_tuple[i][0],
                "product_indices": self.reaction_num_tuple[i][1]
            })
        info = {
            "n_reactions": len(self.reactions_list),
            "n_species": len(self.species_atoms_list),
            "reactions": reactions
        }
        with open(file_name, mode="w") as f:
            json.dump(info, f, indent=2)

        parent_dir = os.path.dirname(file_name)
        intermediate_path = os.path.join(parent_dir, "intermediates.extxyz")
        write(intermediate_path, self.species_atoms_list, format="extxyz")

        return f"Informations of all the reactions are written to {file_name}. The atomic structures of all the intermediates are save in {intermediate_path}."


    def render_system(self, dir_name:str, backend:str):
        n_species = len(self.species_atoms_list)
        n_reactions = len(self.reactions_list)

        n_col = 3 if n_species < 10 else 5
        n_row = int(float(n_species)/n_col)
        if n_row * n_col < n_species:
            n_row += 1
        draw_molecules_array(self.species_atoms_list, 
                             self.species_mols_list,
                             nrow=n_row, ncol=n_col, 
                             width=3*n_col, 
                             height=3*n_row, 
                             save_name=os.path.join(dir_name, "intermediates.png"),
                             dpi=75, backend=backend)
        
        if backend == "ase":
            for i, reac in enumerate(self.reactions_list):
                name = os.path.join(dir_name, f"reaction_{i}.png")
                reac_atoms = reac.reactants_atoms
                reac_coeff = [1] * len(reac_atoms)
                prod_atoms = reac.products_atoms
                prod_coeff = [1] * len(prod_atoms)
                
                visualize_reaction(reactants=reac_atoms, 
                                   reactant_coeffs=reac_coeff,
                                   products=prod_atoms, 
                                   product_coeffs=prod_coeff,
                                   width=6, height=2.5, 
                                   save_name=name, dpi=75)
        else:
            for i, reac in enumerate(self.reactions_list):
                name = os.path.join(dir_name, f"reaction_{i}.png")
                reac_mols = reac.reactants_mols
                reac_coeff = [1] * len(reac_mols)
                prod_mols = reac.products_mols
                prod_coeff = [1] * len(prod_mols)
                visualize_reaction_rdkit(reactants=reac_mols,
                                         reactant_coeffs=reac_coeff,
                                         products=prod_mols,
                                         product_coeffs=prod_coeff,
                                         width=6, height=2.5, 
                                         save_name=name, dpi=75)

        

        return f"Structures of {n_species} intermediates are shown in {os.path.join(dir_name, "intermediates.png")}\n {n_reactions} reactions are shown in pictures reaction_0.png to reaction_{n_reactions-1}.png in {dir_name}."

    def _delete_related_reactions(self, 
                                  initial_reactions_indices:List[int],
                                  initial_species_indices:Set[int]) -> List[int]:
        
        # 不能删除初始反应物相关的反应
        # 提供初始的被删除反应的索引和反应涉及到的物种，据此找出所有需要被删除的反应
        # 例如反应网络里有一条路径 (1) CH + O <-> HCO, (2) HCO <-> CO + H. 同时没有其他
        # 反应生成或消耗HCO，那么删除了(1) 和 (2)中的其中一个必然要删除另外一个。
        reactions_need_tobe_deleted = initial_reactions_indices
        species_in_deleted_reactions = initial_species_indices
        nreactions = len(self.reactions_list)
        while True:
            find_delete = False 
            collect_new_bad_species = set()
            # 检查每个涉及到的中间物种
            for ss in species_in_deleted_reactions:
                reaction_has_ss = []
                # 收集剩余反应里，仍然包含物种ss的反应索引
                for i in range(nreactions):
                    if i in reactions_need_tobe_deleted:
                        continue
                    num_tuple = self.reaction_num_tuple[i]
                    if (ss in num_tuple[0]) or (ss in num_tuple[1]):
                        reaction_has_ss.append(i)
                # 只有一个反应含有ss物种，则该物种无法被生成或消耗
                # 这个反应导向一个死路，所以该反应需要被删除
                if len(reaction_has_ss) == 1: 
                    reactions_need_tobe_deleted.append(reaction_has_ss[0])
                    num_tuple = self.reaction_num_tuple[reaction_has_ss[0]]
                    # 把新反应涉及到的物种加入到 collect_new_bad_species
                    collect_new_bad_species.update(num_tuple[0])
                    collect_new_bad_species.update(num_tuple[1])
                    # 我们找到了新的需要删除的反应，并且更新了需要检查的物种集合
                    find_delete = True 
            # 从新物种集合里去掉反应物和最终产物
            collect_new_bad_species.difference_update(self.reactants_products_indices)
            # 更新该集合
            species_in_deleted_reactions.update(collect_new_bad_species)
            # 所有需要删除的反应已经找到，退出
            if not find_delete:
                break  
        return reactions_need_tobe_deleted
    

    def _update_reactions_and_species(self, 
                                      reactions:List[ReactionHelper], 
                                      has_duplicated:bool = False):
        '''
        Update `self.species_atoms_list`, `self.reaction_num_tuple` and `self.reactions_list`
        '''
        # 先把所有的数据都初始化
        self.species_atoms_list = self.reactants + self.products
        self.species_mols_list = [ase_atoms_to_mol(item) for item in self.species_atoms_list]
        self.reactions_list     = []
        self.reaction_num_tuple = []
        self.reactants_products_indices = list(range(len(self.species_atoms_list)))

        # 现有的物种是反应物和最终产物，先把它们加入 species_smiles 
        self.species_smiles_list = [Chem.MolToSmiles(mol_item) for mol_item in self.species_mols_list ]
        reaction_numstr_set = set()

        for reaction in reactions:
            for i, s in enumerate(reaction.reactants_smiles):
                if not s in self.species_smiles_list:
                    self.species_smiles_list.append(s)
                    self.species_atoms_list.append(reaction.reactants_atoms[i])
                    self.species_mols_list.append(reaction.reactants_mols[i])

            for i, s in enumerate(reaction.products_smiles):  
                if not s in self.species_smiles_list:
                    self.species_smiles_list.append(s)
                    self.species_atoms_list.append(reaction.products_atoms[i])
                    self.species_mols_list.append(reaction.products_mols[i])

            # 现有的物种都已经自动被赋予id，构建反应对应的数字字符串
            ids_of_reactants = sorted([self.species_smiles_list.index(ss) for ss in reaction.reactants_smiles])
            ids_of_products = sorted([self.species_smiles_list.index(ss) for ss in reaction.products_smiles])

            if has_duplicated:
                numstr_reactants = '-'.join([str(a) for a in ids_of_reactants])
                numstr_products = '-'.join([str(a) for a in ids_of_products]) 
                numstr = numstr_reactants +','+ numstr_products if numstr_reactants > numstr_products \
                    else numstr_products +','+ numstr_reactants
                if not numstr in reaction_numstr_set:
                    reaction_numstr_set.add(numstr)
                    self.reactions_list.append(reaction)
                    self.reaction_num_tuple.append((ids_of_reactants, ids_of_products))
            else:
                self.reactions_list.append(reaction)
                self.reaction_num_tuple.append((ids_of_reactants, ids_of_products))
          
    def generate_elementary_steps(self,
                           ntimes:int = -1,
                           max_natoms_tot:int = -1, 
                           min_natoms_tot:int = 0, 
                           max_natoms:Dict[str, int] = None, 
                           min_natoms:Dict[str, int] = None, 
                           forbidden_pattern: List[Tuple[str, str]] = None):
        '''
        Generate reactions. 
        Args:
            ntimes: The number of times the elementary steps is sampled. Here, "sample" means use BFS and DFS to search elementary  steps. If `ntimes > 0`, use BFS firstly to get the shortest path and the remaining attempts will be made using DFS (DFS does not always guarantee to find a path, so multiple attempts are necessary). If `ntimes = -1` (default), a BFS-like method will be used to find all possible elementary steps.
            max_natoms_tot: Maximum number of atoms in the intermediates. `-1` means no limitation.
            min_natoms_tot: Minimum number of atoms in the intermediates.
            max_natoms: A `dict` used to limit the maximum number of atoms of each element in the intermediates. If omitted, a default value is calculated from the final products.
            min_natoms: A `dict` used to limit the minimum number of atoms of each element in the intermediates.
            forbidden_pattern: Substructures of intermediates that are prohibited. For example, use `('H,O,C,O,H','0-1,1-2,2-3,3-4')` to represent geminal diol. If omitted, a list of default patterns are generated.
        '''

        if forbidden_pattern is None:
            forbidden_pattern = _get_default_bad_patterns(self.elements, max_natoms) 
        if max_natoms is None:
            max_natoms = _get_default_max_natoms(self.products)
        
        ern = ExploreReactionNetwork(self.reactants, 
                                     self.products, 
                                     max_natoms_tot, 
                                     min_natoms_tot, 
                                     max_natoms, 
                                     min_natoms, 
                                     forbidden_pattern)
        if ntimes > 0:
            ern.search(bfs=True) # 第一次先用广度优先搜索寻找最短路径
            reactions = ern.get_reactions()
            for i in range(1, ntimes):
                ern.search()
                if ern.final_state is None:
                    continue
                reactions.extend(ern.get_reactions())
        else:
            ern.run()
            reactions = ern.get_reactions()

        self._update_reactions_and_species(reactions, ntimes > 0)
        
    def delete_species(self, idx:int):
        '''
        Delete the species of `idx` and the related elementary steps.
        '''
        nreactions = len(self.reaction_num_tuple)
        reactions_need_tobe_deleted = []
        species_in_deleted_reactions = set()

        # 先遍历每个反应，如果反应中包含物种idx，则该反应需要被删除
        for i in range(nreactions):
            num_tuple = self.reaction_num_tuple[i]
            if (idx in num_tuple[0]) or (idx in num_tuple[1]):
                reactions_need_tobe_deleted.append(i)
                # 同时记录该反应涉及到的其他物种
                species_in_deleted_reactions.update(num_tuple[0])
                species_in_deleted_reactions.update(num_tuple[1])
        
        # 从这些物种里去掉反应物和最终产物
        species_in_deleted_reactions.difference_update(self.reactants_products_indices )
        reactions_need_tobe_deleted = self._delete_related_reactions(reactions_need_tobe_deleted,
                                                                     species_in_deleted_reactions)
        reactions = [self.reactions_list[i] for i in range(nreactions) if not i in reactions_need_tobe_deleted]
        self._update_reactions_and_species(reactions)


    def delete_reaction(self, idx:int):
        '''
        Delete an elementary reaction.
        '''
        nreactions = len(self.reaction_num_tuple)
        reactions_need_tobe_deleted = [idx]
        species_in_deleted_reactions = set(self.reaction_num_tuple[idx][0])
        species_in_deleted_reactions.update(self.reaction_num_tuple[idx][1])
        species_in_deleted_reactions.difference_update(self.reactants_products_indices)
        reactions_need_tobe_deleted = self._delete_related_reactions(reactions_need_tobe_deleted,
                                                                     species_in_deleted_reactions)
        reactions = [self.reactions_list[i] for i in range(nreactions) if not i in reactions_need_tobe_deleted]
        self._update_reactions_and_species(reactions)
    

    def merge_reactions(self, id1:int, id2:int):
        '''
        Merge elementary reactions.  `generate_elementary_steps` can only generate reactions which only change a single chemical bond. Reactions such as atomic group transfer and insertion will be split into multiple steps. For example: reactions NH2OH + H <-> NH2 + H2O will be (1) NH2OH <-> NH2 + OH, (2) H + OH <-> H2O, reaction CH3CH3 + CH2: <-> CH3CH2CH3 will be (1) CH3CH3 <-> CH3 + CH3 (2)CH3 + :CH2 <-> CH3CH2 (3)CH3CH2 + CH3 <-> CH3CH2CH3. 
        '''
        # (0) NOH <-> N + OH
        # (1) NH + H <-> NH2
        # (2) NHOH <-> NH + OH
        # (3) NH2OH <-> NH2 + OH
        # (4) OH + H <-> H2O
        # 对于上面的四个反应，如果要合并(3)和(4)，则应该去掉(3) 保留 (4)。虽然去掉(4)之后(0)和(2)仍然能构成OH的产生消耗路径
        # 但此时反应网络的动力学已经不对了，因为OH不可能是以这个方式产生和消耗的。因此该函数不能主动决定合并之后哪个子反应可以被删除
        # 另外两个可以合并的反应一定是一个慢反应（决速步）一个快反应（快速平衡），合并之后的总反应跟慢反应可以视作同一种。
        reac1 = self.reactions_list[id1]
        reac2 = self.reactions_list[id2]
        try:
            reac_new = reac1.mergeReaction(reac2)
        except ValueError as e:
            raise ValueError(f"In reaction {id2}, more than one bond are changed.")

        if reac_new is None:
            raise ValueError(f"Can not merge reactions {id1} and {id2}, the result reaction will not be an elementary step.")
        
        reactions = self.reactions_list
        reactions.append(reac_new)
        self._update_reactions_and_species(reactions)
    

    def bind_to_surfaces(self, surfaces:List[Atoms]) -> list:
        # 获取表面的最表层原子 
        surface_info_list = []

        for surface in surfaces:
            surface_sites, dist_mat = get_outmost_atoms_for_surface(surface)
            gcn_list = get_gcn(surface, surface_sites)
            sites_groups = [] # 根据gcn对位点进行分组
            # 把位点按照广义配位数进行分类，每一类中，类与类之间是一对。如果有两个不同吸附物还有吸附物的组合
            for idx, gcn in zip(surface_sites, gcn_list):
                for group in sites_groups:
                    if abs(gcn - gcn_list[surface_sites.index(group[0])]) < 0.1:
                        group.append(idx)
                        break 
                else:
                    sites_groups.append([idx])
            
            surface_info_list.append({
                "surface": surface,
                "dist_mat": dist_mat, 
                "sites_groups": sites_groups,
            })

        # 对齐初末态，获得结合位点
        initial_final_info_list = []
        for reac_helper in self.reactions_list:
            ini_smiles_list = reac_helper.reactants_smiles
            fin_smiles_list = reac_helper.products_smiles

            mapping_index, reactants_combined, products_combined = reac_helper.get_aligned_reactants_and_products()
            total_atoms = reactants_combined + products_combined
            total_atoms_radius = np.max(total_atoms.get_all_distances(mic=True))*0.5 \
                    + np.max([ covalent_radii[a] for a in reactants_combined.get_atomic_numbers() ])
            total_atoms_center = np.mean(total_atoms.positions, axis=0)


            reactants_binding_index = []
            products_binding_index = []
            start = 0
            for frag in reac_helper.reactants_atoms:
                bidx = get_adsorbate_binding_atom_index(frag)
                if not bidx is None:
                    reactants_binding_index.append([int(bid+start) for bid in bidx])
                else:
                    reactants_binding_index.append(None)
                start += len(frag)
            
            for frag in reac_helper.products_atoms:
                bidx = get_adsorbate_binding_atom_index(frag)
                if not bidx is None:
                    products_binding_index.append([int(bid+start) for bid in bidx])
                else:
                    products_binding_index.append(None)
                start += len(frag)
            

            nfrag_ini = len(reactants_binding_index)
            nfrag_fin = len(products_binding_index)
        
            # 如果有两个片段，则这两个片段必定有分别有可结合的原子，其中任何一个片段都不可能是饱和的
            is_symmetric = False
            if nfrag_fin == 2: 
                if fin_smiles_list[0] == fin_smiles_list[1]:
                    is_symmetric = True  
                total_binding_index = products_binding_index
            elif nfrag_ini == 2:
                if ini_smiles_list[0] == ini_smiles_list[1]:
                    is_symmetric = True  
                total_binding_index = reactants_binding_index
            else: # 初末态都只有一个片段
                total_binding_index = []
                for item in reactants_binding_index+products_binding_index:
                    if not item is None:
                        total_binding_index.extend(item)
                total_binding_index = [total_binding_index]
            
            initial_final_info_list.append({
                "is_symmetric": is_symmetric,
                "total_atoms": total_atoms,
                "total_atoms_radius": total_atoms_radius,
                "total_atoms_center": total_atoms_center,
                "total_binding_index": total_binding_index,
                "mapping_index": mapping_index
            })


        collect_result = []

        for surf_info in surface_info_list:

            n_surface_atoms = len(surf_info["surface"])

            intermediate_index_dict = dict()
            reactions_result_list = []

            for ir, ini_fin_info in enumerate(initial_final_info_list):
                ini_result, fin_result, ads_index = _bind_ini_fin_to_surface(total_atoms=ini_fin_info["total_atoms"],
                                                                  total_atoms_radius=ini_fin_info["total_atoms_radius"],
                                                                  total_atoms_center=ini_fin_info["total_atoms_center"],
                                                                  total_binding_index=ini_fin_info["total_binding_index"],
                                                                  mapping_index=ini_fin_info["mapping_index"],
                                                                  surface=surf_info["surface"],
                                                                  dist_mat=surf_info["dist_mat"],
                                                                  sites_groups=surf_info["sites_groups"],
                                                                  is_symmetric=ini_fin_info["is_symmetric"])
                reac = self.reactions_list[ir]
                if len(reac.reactants_smiles) == 1:
                    if not reac.reactants_smiles[0] in intermediate_index_dict:
                        intermediate_index_dict[reac.reactants_smiles[0]] = (ir, 0, ads_index)
                elif len(reac.products_smiles) == 1:
                    if not reac.products_smiles[0] in intermediate_index_dict:
                        intermediate_index_dict[reac.products_smiles[0]] = (ir, 1, ads_index)

                reactions_result_list.append({
                    "name": self.reactions_list[ir].unique_reaction_str,
                    "ini": ini_result,
                    "fin": fin_result,
                    "ads_atoms_index": list(range(n_surface_atoms, len(ini_result[0])))
                })

            collect_result.append((intermediate_index_dict, reactions_result_list))

        return collect_result 
                

            
