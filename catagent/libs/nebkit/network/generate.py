from __future__ import annotations

from typing import List, Tuple, Dict, Set, Any

from ase import Atoms 
from ase.data import atomic_numbers
from ase.io import write 
 
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import rdmolops, CombineMols

from collections import deque
from random import shuffle

import numpy as np

from .helper import ReactionHelper
from catagent.libs.nebkit.tools.utils import ase_atoms_to_mol, pattern_to_mol, mol_to_ase_atoms

MAX_COORD_NUM = {1: 1, # H
                 6: 4, # C
                 7: 3, # N, 4配位的有NH4+和季铵盐
                 8: 2, # O
                 9: 1, # F
                 17: 1 } #Cl

class SingleBondReaction:
    '''
    只发生一个化学键变化的反应。
    '''
    def __init__(self, 
                 reactants_str:List[str], 
                 products_str:List[str],
                 mapping_atom_index: Tuple[np.ndarray, np.ndarray]):
        '''
        Args:
            reactants_str: 反应物的smiles字符串列表
            products_str: 产物的smiles字符串列表
            mapping_atom_index: 表示反应物和产物原子映射关系的两个元素的tuple，这两个元素是等长的一维整数数组
        '''
        
        self.reactants_str = reactants_str
        self.products_str = products_str
        self.reactants_mol = None 
        self.products_mol = None
        self.mapping_atom_index = mapping_atom_index
        
        self.rstr = " + ".join(sorted(self.reactants_str))
        self.pstr = " + ".join(sorted(self.products_str))
        self.reaction_str = self.rstr + " <-> " + self.pstr
    
    def set_mols(self, reac:List[Chem.Mol], prod:List[Chem.Mol]):
        self.reactants_mol = reac 
        self.products_mol= prod 
    
    def __eq__(self, value:SingleBondReaction): #两个反应是否相等？
        if self.rstr == value.rstr and self.pstr == value.pstr:
            return True 
        if self.rstr == value.pstr and self.pstr == value.rstr:
            return True 
        return False 
    def __repr__(self):
        return self.reaction_str
    def __str__(self):
        return self.reaction_str


def count_atoms(mol:Chem.Mol):
    result = dict()
    for atom in mol.GetAtoms():
        n = atom.GetAtomicNum()
        if n in result:
            result[n] += 1
        else:
            result[n] = 1
    return result  

def make_pure_graph(mol: Chem.Mol) -> Chem.Mol:
    """Force the molecule to be a pure connectivity graph (no radicals, no implicit H)."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
        atom.SetFormalCharge(0)   # uncomment if you also want to ignore charges
    rw.UpdatePropertyCache(strict=False)
    return rw.GetMol()


def merge_two_mols(mol1:Chem.Mol, index1:int, mol2:Chem.Mol, index2:int) -> Chem.Mol:
    combined = CombineMols(mol1, mol2)
    combined_rw = Chem.RWMol(combined) 
    mol1_atoms = mol1.GetNumAtoms()
    new_id2 = index2 + mol1_atoms
    combined_rw.AddBond(index1, new_id2, Chem.BondType.SINGLE)
    return combined_rw.GetMol()

def form_bond_in_mol(mol:Chem.Mol, index1:int, index2:int) -> Chem.Mol:
    new_mol = Chem.RWMol(mol)
    new_mol.AddBond(index1, index2, Chem.BondType.SINGLE)
    new_mol = new_mol.GetMol()
    return new_mol

def break_bond_in_mol(mol:Chem.Mol, index1:int, index2:int)->Tuple[Chem.Mol, ...]:
    bond = mol.GetBondBetweenAtoms(index1, index2)
    if bond is None:
        raise ValueError(f"There is no bond between {index1} and {index2}")    
    rw_mol = Chem.RWMol(mol)
    rw_mol.RemoveBond(index1, index2)
    temp_mol = rw_mol.GetMol()
    frags = Chem.GetMolFrags(temp_mol, asMols=True)
    # Removing a bond that belongs to a ring does not disconnect the molecule,
    # so `frags` may contain a single fragment (ring opening). Return however
    # many fragments result instead of assuming exactly two.
    return tuple(make_pure_graph(frag) for frag in frags)


def get_mol_info(mol:Chem.Mol):
    natoms = mol.GetNumAtoms()
    atomic_numbers = []
    coord_numbers = []
    unsaturated_index = []
    for i, atom in enumerate(mol.GetAtoms()):
        an = atom.GetAtomicNum()
        cn = atom.GetDegree()
        if cn < MAX_COORD_NUM[an]:
            unsaturated_index.append(i)
        atomic_numbers.append(an)
        coord_numbers.append(cn)
        
    an_dict = dict()
    for an in atomic_numbers:
        if an in an_dict:
            an_dict[an] += 1
        else:
            an_dict[an] = 1
    
    info = {
        "unsaturated_index": unsaturated_index,
        "atomic_numbers":atomic_numbers,
        "natoms": natoms, 
        "ele_nums": an_dict,
        "coord_nums": coord_numbers,
        "adj_matrix": rdmolops.GetAdjacencyMatrix(mol)
    }
    return info 

class ExploreReactionNetwork:
    '''
    给定反应物和产物探索可能的基元反应。只考虑 H,C,N,O,F,Cl 6种元素
    '''
    def __init__(self,
                 reactants:List[Atoms], 
                 products:List[Atoms],
                 max_natoms_tot:int = -1,
                 min_natoms_tot:int = 0,
                 max_natoms:Dict[str, int] = None,
                 min_natoms:Dict[str, int] = None,
                 forbidden_pattern: List[Tuple[str, str]] = None):
        
        self.max_natoms_tot = max_natoms_tot if max_natoms_tot > 0 else 100
        self.min_natoms_tot = min_natoms_tot

        if not max_natoms is None:
            self.max_natoms = { atomic_numbers[key]:max_natoms[key] for key in max_natoms.keys()}
        else:
            self.max_natoms = dict()

        if not min_natoms is None:
            self.min_natoms = {atomic_numbers[key]:min_natoms[key] for key in min_natoms.keys()}
        else:
            self.min_natoms = dict()

        self.reac_mols = [ase_atoms_to_mol(atoms) for atoms in reactants]
        self.prod_mols = [ase_atoms_to_mol(atoms) for atoms in products]

        self.prod_smiles = {Chem.MolToSmiles(mol, canonical=True) for mol in self.prod_mols}

        self.forbidden_patterns = []
        if forbidden_pattern:
            for symbols, adjacent_strs in forbidden_pattern:
                self.forbidden_patterns.append(pattern_to_mol(symbols, adjacent_strs))

        self.reaction_str_list = []
        self.mols_list = deepcopy(self.reac_mols)
        self.mols_smiles_list = []
        self.mols_info_list = []
        for item in self.mols_list:
            self.mols_smiles_list.append(Chem.MolToSmiles(item, canonical=True))
            self.mols_info_list.append(get_mol_info(item))
        
        self.final_state = None
    
    def update_state(self,
                    state:Dict[str, Any],
                    rstr:List[str],
                    rid:List[int], 
                    pstr:List[str], 
                    pmol:List[Chem.Mol],
                    mapping_atom_index:Tuple[np.ndarray, np.ndarray])->None | Dict[str, Any]:
        '''
        辅助函数，用于更新`self.reaction_str_list`, `self.mols_list`, `self.mols_smiles_list`和`self.mols_info_list`
        根据情况返回新的正确的`current_mols` 和 `previous_mols`。
        Args:
            state: 记录了当前状态的字典
            rstr: 反应物SMILES字符串的列表
            rid:  反应物在self.mols_list中的索引
            pstr: 产物SMILES字符串的列表
            pmol: 产物分子的列表
            mapping_atom_index: 表示原子映射关系的tuple

        '''
        # 反应是否已经存在？如果反应存在则返回None
        reac_str = SingleBondReaction(rstr, pstr, mapping_atom_index)
        for ri in state["reaction_index"]:
            if reac_str == self.reaction_str_list[ri]:
                return None 
        # 产物分子是否存在禁止的模式？如果存在这种模式则反应不可发生
        for p_mol in pmol:
            for bad_pattern in self.forbidden_patterns:
                if p_mol.HasSubstructMatch(bad_pattern):
                    return None
        # 反应没有禁止的模式，所以可以发生，检查反应是否在列表中存在
        if reac_str in self.reaction_str_list: #如果已经在反应列表存在，则获取id
            reac_id = self.reaction_str_list.index(reac_str)
        else:
            reac_str.set_mols([self.mols_list[r_id] for r_id in rid], pmol)

            self.reaction_str_list.append(reac_str) #如果没在反应列表中，则加入同时获得它的id
            reac_id = len(self.reaction_str_list) - 1
        
        new_reaction_index = state["reaction_index"].copy()
        new_reaction_index.add(reac_id)

        new_current_mols   = state["current_mols"].copy()
        new_previous_mols  = state["previous_mols"].copy()

        for r_id in rid: #把反应掉的分子都移动到previous_mols
            if r_id in new_current_mols:
                new_current_mols.remove(r_id)
                new_previous_mols.add(r_id)
        
        for i, p_mol in enumerate(pmol):
            # 如果是一个没有存在过的分子，则把它加入分子列表
            if not pstr[i] in self.mols_smiles_list:
                self.mols_list.append(p_mol)
                self.mols_smiles_list.append(pstr[i])
                self.mols_info_list.append(get_mol_info(p_mol))
                new_current_id = len(self.mols_list) - 1 
            else:
                new_current_id = self.mols_smiles_list.index(pstr[i])
            
            if not new_current_id in new_previous_mols:
                new_current_mols.add(new_current_id)
        
        return {"current_mols": new_current_mols, 
                "previous_mols": new_previous_mols, 
                "reaction_index":new_reaction_index,
                "depth": state["depth"]+1}

    def get_next_states(self, state:Dict[str, Any]):
        results = []
        # print("=========== step1 ===========")
        # 1. 进行current_mols 的分子内成键操作
        for i in state["current_mols"]:
            mol = self.mols_list[i]
            mols_info = self.mols_info_list[i]
            smiles = self.mols_smiles_list[i]
            index_can_bond = mols_info["unsaturated_index"]
            n_index_can_bond = len(index_can_bond)

            for i1 in range(n_index_can_bond):
                idx1 = index_can_bond[i1]
                for i2 in range(i1+1, n_index_can_bond):
                    idx2 = index_can_bond[i2]
                    if mols_info["adj_matrix"][idx1, idx2] == 1: #因为只考虑单键，所以已经相连的原子对不考虑
                        continue 
                    new_mol = form_bond_in_mol(mol, idx1, idx2)
                    new_smiles = Chem.MolToSmiles(new_mol, canonical=True)

                    mapping_atom_index = (np.arange(mols_info["natoms"]), np.arange(mols_info["natoms"]))
                    # active_atom_index = [[idx1, idx2]] 
                    # reaction_type = ReactionType.IN_MOL_FORM_BOND

                    new_state = self.update_state(state, [smiles], [i], [new_smiles], [new_mol], mapping_atom_index)

                    if (not new_state is None) and (not new_state in results):
                        results.append(new_state)
        
        # print("=========== step2 ===========")
        # 2. 进行 current_mols的 分子之间的成键操作
        n_current = len(state["current_mols"])
        li_current_mols = list(state["current_mols"])
        for i in range(n_current):
            idx1             = li_current_mols[i]
            mol_1            = self.mols_list[idx1]
            mols_info_1      = self.mols_info_list[idx1]
            smiles_1         = self.mols_smiles_list[idx1]
            index_can_bond_1 = mols_info_1["unsaturated_index"]
            for j in range(i, n_current):
                idx2             = li_current_mols[j]
                mol_2            = self.mols_list[idx2]
                mols_info_2      = self.mols_info_list[idx2]
                smiles_2         = self.mols_smiles_list[idx2]
                index_can_bond_2 = mols_info_2["unsaturated_index"]
                # 是否超过总原子数？
                if mols_info_1["natoms"] + mols_info_2["natoms"] > self.max_natoms_tot:
                    continue 
                # 是否超过每种原子的数目？
                exceed = False
                for key in self.max_natoms.keys():
                    ne_1 = mols_info_1["ele_nums"][key] if key in mols_info_1["ele_nums"] else 0
                    ne_2 = mols_info_2["ele_nums"][key] if key in mols_info_2["ele_nums"] else 0 
                    if ne_1 + ne_2 > self.max_natoms[key]:
                        exceed = True 
                        break 
                if exceed:
                    continue
                for atom_idx1 in index_can_bond_1:
                    for atom_idx2 in index_can_bond_2:
                        new_mol = merge_two_mols(mol_1, atom_idx1, mol_2, atom_idx2)
                        new_smiles = Chem.MolToSmiles(new_mol, canonical=True)

                        mapping_atom_index = (np.hstack((np.arange(mols_info_1["natoms"]), np.arange(mols_info_2["natoms"]))), 
                                              np.arange(mols_info_1["natoms"]+ mols_info_2["natoms"]))
                        # active_atom_index = [[atom_idx1], [atom_idx2]]
                        # reaction_type = ReactionType.BETWEEN_MOLS_FORM_BOND
                        new_state = self.update_state(state, [smiles_1, smiles_2], [idx1, idx2], [new_smiles], [new_mol], mapping_atom_index)

                        if (not new_state is None) and (not new_state in results):
                            results.append(new_state)
        
        # print("=========== step3 ===========")
        # 3. 进行current_mols 和 previous_mols的分子间成键操作
        for i in state["current_mols"]:
            mol_i = self.mols_list[i]
            mols_info_i = self.mols_info_list[i]
            smiles_i = self.mols_smiles_list[i]
            index_can_bond_i = mols_info_i["unsaturated_index"]
            for j in state["previous_mols"]:
                mol_j = self.mols_list[j]
                mols_info_j = self.mols_info_list[j]
                smiles_j = self.mols_smiles_list[j]
                index_can_bond_j = mols_info_j["unsaturated_index"]
                # 是否超过总原子数？
                if mols_info_i["natoms"] + mols_info_j["natoms"] > self.max_natoms_tot:
                    continue
                # 是否超过每种原子的数目？
                exceed = False
                for key in self.max_natoms.keys():
                    ne_i = mols_info_i["ele_nums"][key] if key in mols_info_i["ele_nums"] else 0
                    ne_j = mols_info_j["ele_nums"][key] if key in mols_info_j["ele_nums"] else 0 
                    if ne_i + ne_j > self.max_natoms[key]:
                        exceed = True 
                        break 
                if exceed:
                    continue
                # 检查每一种成键组合
                for ii in index_can_bond_i:
                    for jj in index_can_bond_j:
                        new_mol = merge_two_mols(mol_i, ii, mol_j, jj)
                        new_smiles = Chem.MolToSmiles(new_mol, canonical=True)

                        mapping_atom_index = (np.hstack((np.arange(mols_info_i["natoms"]), np.arange(mols_info_j["natoms"]))), 
                                              np.arange(mols_info_i["natoms"]+ mols_info_j["natoms"]))
                        # active_atom_index = [[ii], [jj]]
                        # reaction_type = ReactionType.BETWEEN_MOLS_FORM_BOND

                        new_state = self.update_state(state, [smiles_i, smiles_j], [i, j], [new_smiles], [new_mol], mapping_atom_index)

                        if (not new_state is None) and (not new_state in results):
                            results.append(new_state) 
         
        # print("=========== step4 ===========")
        # 4. 进行分子内断键操作
        for i in state["current_mols"]:
            mol = self.mols_list[i]
            mols_info = self.mols_info_list[i]
            smiles = self.mols_smiles_list[i]

            bonds_list = []
            adj_matrix = mols_info["adj_matrix"]
            for i_row in range(adj_matrix.shape[0]):
                for i_col in range(i_row+1, adj_matrix.shape[1]):
                    if adj_matrix[i_row, i_col] == 1:
                        bonds_list.append((i_row, i_col))
            
            for a, b in bonds_list:
                new_mol = list(break_bond_in_mol(mol, a, b))
                new_smiles = [Chem.MolToSmiles(fragment, canonical=True) for fragment in new_mol]

                #active_atom_index = [[a, b]] 
                if len(new_smiles) == 1:
                    mapping_atom_index = (np.arange(mols_info["natoms"]),  np.arange(mols_info["natoms"]))
                    #reaction_type = ReactionType.MOL_BREAK_BOND_1
                else:
                    mapping_atom_index = (np.arange(mols_info["natoms"]), 
                                          np.hstack((np.arange(new_mol[0].GetNumAtoms()), np.arange(new_mol[1].GetNumAtoms()))))
                    #reaction_type = ReactionType.MOL_BREAK_BOND_2
                    #如果产生了两个分子片，检测是否小于最小原子数
                    if new_mol[0].GetNumAtoms() < self.min_natoms_tot or new_mol[1].GetNumAtoms() < self.min_natoms_tot:
                        continue 
                    # 分别检测每个分子片中各种原子数是否小于规定值
                    bad_step = False 
                    count = count_atoms(new_mol[0])
                    for key in count.keys():
                        if key in self.min_natoms and count[key] < self.min_natoms[key]:
                            bad_step = True 
                            break 
                    if bad_step:
                        continue
                    count = count_atoms(new_mol[1])
                    for key in count.keys():
                        if key in self.min_natoms and count[key] < self.min_natoms[key]:
                            bad_step = True 
                            break 
                    if bad_step:
                        continue
                    
                new_state = self.update_state(state, [smiles], [i], new_smiles, new_mol, mapping_atom_index)

                if (not new_state is None) and (not new_state in results):
                    results.append(new_state)        
        return results


    def search(self, 
            random = True,
            bfs = False, 
            n_max_step = 100000,
            n_stat = 2000,
            n_wait = 0,
            max_n_fail = 1000):
        
        state = {
            "current_mols":set(range(len(self.reac_mols))),
            "previous_mols":set(),
            "reaction_index":set(),
            "depth": 0
        }

        queue = deque([state])

        n_step = 0
        n_fail = 0
        i_wait = 0
        stat_dict = dict()

        while queue and n_step < n_max_step:
            n_step += 1
            if bfs:
                state = queue.popleft()
            else: 
                state = queue.pop() # 只有DFS才统计栈深度
                if n_stat > 0 and (n_step+1)%n_stat == 0: 
                    if n_fail >= max_n_fail:
                        i_wait += 1
                        if i_wait > n_wait:
                            top_level = min(list(stat_dict.keys()))
                            while queue and state["depth"] > top_level:
                                state = queue.pop()
                            i_wait = 0
                    n_fail = 0
                    stat_dict = dict()
                
                if n_stat > 0:
                    depth = state["depth"] #记录栈的访问深度
                    if depth in stat_dict:
                        stat_dict[depth] += 1
                    else:
                        stat_dict[depth] = 1
            
            next_states = self.get_next_states(state)
            n_states = len(next_states)
            if n_states == 0:
                # 没有产生新状态，说明current_mols中的分子无法断键（因为形成该分子的反应已经被包含在内）
                # 也无法成键（因为所有原子都已经饱和）。这说明current_mols中全部都是新的饱和产物分子。
                # 把它们都移入 previous_mols
                new_previous_mols = state["previous_mols"].copy()
                new_previous_mols.update(state["current_mols"])
                # 检测是否达到终止条件
                previous_smiles = {self.mols_smiles_list[idx] for idx in new_previous_mols}
                if previous_smiles.issuperset(self.prod_smiles):
                    self.final_state = {"current_mols":set(), "previous_mols":new_previous_mols,
                                        "reaction_index":state["reaction_index"].copy(),
                                        "depth": state["depth"]}
                    return True
                else:
                    n_fail += 1
                    continue

            index = list(range(n_states))
            if random and (not bfs): #随机搜索对bfs没有意义
                shuffle(index)
            n_empty_and_fail = 0
            for ii in index:
                new_state = next_states[ii]
                # 如果 new_current_mols 有产物分子，则把它移入 new_previous_mols
                for idx in list(new_state["current_mols"]):
                    if self.mols_smiles_list[idx] in self.prod_smiles:
                        new_state["current_mols"].remove(idx)
                        new_state["previous_mols"].add(idx)
                # 终止条件：new_current_mols 为空，产物分子全部在 new_previous_mols 中
                if len(new_state["current_mols"]) == 0:
                    previous_smiles = {self.mols_smiles_list[idx] for idx in new_state["previous_mols"]}
                    if previous_smiles.issuperset(self.prod_smiles):
                        self.final_state = new_state
                        return True 
                    else:
                        n_empty_and_fail += 1 # 失败的搜索
                else: 
                    queue.append(new_state) # 不满足终止条件则入队列
            if n_empty_and_fail == n_states:
                n_fail += 1


        return False
    
    def run(self):
        state = {
            "current_mols":set(range(len(self.reac_mols))),
            "previous_mols":set(),
            "reaction_index":set(),
            "depth": 0
        }
        
        while True:
            # aaa = input("aaa:")
            # print('self.mols_smiles_list = ',  self.mols_smiles_list)
            # print('self.reaction_str_list = ', self.reaction_str_list)
            # print('state = ', state)

            next_states = self.get_next_states(state)
            # print('next_states = ', next_states)
            # print("current_mols = ", state["current_mols"])
            
            if len(next_states) == 0:
                # 没有产生新状态，说明current_mols中的分子无法断键（因为形成该分子的反应已经被包含在内）
                # 也无法成键（因为所有原子都已经饱和）。这说明current_mols中全部都是新的饱和产物分子。
                # 把它们都移入 previous_mols
                new_previous_mols = state["previous_mols"].copy()
                new_previous_mols.update(state["current_mols"])
                # 检测是否达到终止条件
                previous_smiles = {self.mols_smiles_list[idx] for idx in new_previous_mols}
                if previous_smiles.issuperset(self.prod_smiles):
                    self.final_state = {"current_mols":set(), "previous_mols":new_previous_mols,
                                        "reaction_index":state["reaction_index"].copy(),
                                        "depth": state["depth"]}
                    return True
                else:
                    self.final_state = None 
                    return False 
            
            

            # 将所有的next_states合并
            new_current_mols = set()
            new_previous_mols = set()
            new_reaction_index = set()
            for ns in next_states:
                new_current_mols.update(ns["current_mols"])
                new_previous_mols.update(ns["previous_mols"])
                new_reaction_index.update(ns["reaction_index"])
            new_current_mols = new_current_mols.difference(new_previous_mols)
            # 如果 new_current_mols 有产物分子，则把它移入 new_previous_mols
            for idx in list(new_current_mols):
                if self.mols_smiles_list[idx] in self.prod_smiles:
                    new_current_mols.remove(idx)
                    new_previous_mols.add(idx)
            
            state = {
                "current_mols": new_current_mols,
                "previous_mols": new_previous_mols,
                "reaction_index": new_reaction_index,
                "depth": next_states[0]["depth"]
            }

            # 终止条件：new_current_mols 为空，产物分子全部在 new_previous_mols 中
            if len(new_current_mols) == 0:
                previous_smiles = {self.mols_smiles_list[idx] for idx in new_previous_mols}
                if previous_smiles.issuperset(self.prod_smiles):
                    self.final_state = state
                    return True 
                else:
                    self.final_state = None 
                    return False 
           
    
    def get_reactions(self)->List[ReactionHelper]:
        '''
        输出最后的基元反应和涉及到的中间产物。
        '''
        if self.final_state is None:
            raise RuntimeError("Please use .run() to generate reations first.")
        
        reactions_list = []
        for i in self.final_state["reaction_index"]:
            reac_str = self.reaction_str_list[i]

            n_atoms = len(reac_str.mapping_atom_index[0])
            mapping_atom_index = np.zeros((3, n_atoms), dtype=int)
            # 原子在反应物中的id，例如对于[CH3, CO]，依次为[0,1,2,3,0,1]
            mapping_atom_index[0, :] = reac_str.mapping_atom_index[0] 
            # 该原子在第几个分子中？
            current_id = -1
            for ii, atom_id in enumerate(reac_str.mapping_atom_index[1]):
                if atom_id == 0:
                    current_id += 1
                mapping_atom_index[1, ii] = current_id
            # 该原子在产物分子中的编号。注意因为目前是用ExploreReactionNetwork类生成的
            # 基元反应，所以原子编号有这样简单的对应。
            mapping_atom_index[2, :] = reac_str.mapping_atom_index[1]

            reac_helper = ReactionHelper(
                reactants_mols=reac_str.reactants_mol,
                products_mols=reac_str.products_mol,
                mapping_atom_index=mapping_atom_index)
                        
            reactions_list.append(reac_helper)
        return reactions_list