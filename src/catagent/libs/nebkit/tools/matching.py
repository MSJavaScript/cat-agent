from rdkit import Chem 
from rdkit.Chem import rdFMCS

from ase.formula import Formula
from ase import Atoms 
from ase.neighborlist import NeighborList, natural_cutoffs

import numpy as np
from math import factorial
from typing import Tuple, List 
from itertools import permutations, product, chain
from copy import deepcopy

from .utils import ase_atoms_to_mol, split_ase_atoms_fragments

def _getall_matches_MCS(mol1:Chem.Mol, heavy_atom_indices1:np.ndarray, 
                        mol2:Chem.Mol, heavy_atom_indices2:np.ndarray)->List[List[Tuple[int, int]]]:
    '''
    Helper function to get the atom indices mapping of molecular scaffold by finding the Maximum 
    Common Substructure (MCS).  This function fails for high symmetry case. For example, the reaction 
    HO-CH2-CH <-> CH2-CH-OH has the same molecular scaffolds for reactant and product (CCO), match the 
    MCS will result in incorrect atom indices mapping.
    Args:
        mol1: Molecular scaffold of reactants, that is, all H atoms are removed from the molecule. 
        heavy_atom_indices1: Indices of heavy atoms in the original reactant molecule.
        mol2: Molecular scaffold of products.
        heavy_atom_indices2: Indices of heavy atoms in the original product molecule.
    Returns:
        All the atom indices mapping, each one is a list of tuple.
    '''
    result = []
    stack = [(mol1, mol2, heavy_atom_indices1.tolist(), heavy_atom_indices2.tolist(), [])]
    while stack:
        node = stack.pop()
        natoms = node[0].GetNumAtoms()
        assert natoms == node[1].GetNumAtoms()

        if natoms == 0:
            total_matches = node[4]
            for item in product(*total_matches):
                index0 = []
                index1 = []
                for a, b in item:
                    index0.extend(a)
                    index1.extend(b)
                
                result.append([(i0,i1) for i0,i1 in zip(index0, index1)])
            continue

        mcs = rdFMCS.FindMCS([node[0], node[1]])
        if mcs.numAtoms == 0: 
            continue  # 这是一个失败的匹配
            
        mol_mcs = Chem.MolFromSmarts(mcs.smartsString)
        match0 = node[0].GetSubstructMatches(mol_mcs, uniquify=False)
        match1 = node[1].GetSubstructMatches(mol_mcs, uniquify=False)
        match_dict0 = dict() #根据匹配的原子进行分组
        for i,item in enumerate(match0):
            sorted_item = tuple(sorted(item, reverse=True))
            if sorted_item in match_dict0:
                match_dict0[sorted_item].append(i)
            else:
                match_dict0[sorted_item] = [i]
        
        match_dict1 = dict() #根据匹配的原子进行分组
        for i,item in enumerate(match1):
            sorted_item = tuple(sorted(item, reverse=True))
            if sorted_item in match_dict1:
                match_dict1[sorted_item].append(i)
            else:
                match_dict1[sorted_item] = [i]
        mol_remain_part0 = []
        index_remain0 = []
        index_matched0 = []

        mol_remain_part1 = []
        index_remain1 = []
        index_matched1 = []

        for item in match_dict0.keys():
            index_matched0.append(item)
            index0 = node[2].copy()
            rw0 = Chem.RWMol(node[0])
            for idx in item:
                index0.pop(idx)
                rw0.RemoveAtom(idx)
            index_remain0.append(index0)
            mol_remain_part0.append(rw0.GetMol())

        for item in match_dict1.keys():
            index_matched1.append(item)
            index1 = node[3].copy()
            rw1 = Chem.RWMol(node[1])
            for idx in item:
                index1.pop(idx)
                rw1.RemoveAtom(idx)
            index_remain1.append(index1)
            mol_remain_part1.append(rw1.GetMol())
        
        for i, mol_i in enumerate(mol_remain_part0):
            for j, mol_j in enumerate(mol_remain_part1):
                part = []
                for i0 in match_dict0[index_matched0[i]]:
                    idx0 = [node[2][k] for k in match0[i0]]
                    for i1 in match_dict1[index_matched1[j]]:
                        idx1 = [node[3][k] for k in match1[i1]]
                        part.append((idx0, idx1))

                macthed_index = deepcopy(node[4])
                macthed_index.append(part)
                stack.append((mol_i, mol_j, index_remain0[i], index_remain1[j], macthed_index))    
    return result

def _getall_matches(fragments1:List[Chem.Mol], mapping_list1:List[tuple],
                    fragments2:List[Chem.Mol], mapping_list2:List[tuple]):
    '''
    Helper function to get the atom indices mapping of molecular scaffold by finding the substructure. This function considers
    three cases:
    - Intramolecular reaction, in this case there is only one fragment in reactant and product. For example, CH2CH2CH2CH2 <-> cyclobutane.
    - Dissociation and coupling reaction. For example, CH3CH2 <-> CH3 + CH2
    - Atomic group migration between two molecules, in this case, there are two fragments in reactant and product. For example, CH3CHOH + CH2 <-> CH3CH + CH2OH
    Args:
        fragments1: A list of fragments for reactant (all H atoms are removed).
        mapping_list1: Atom indices of every heavy atoms in the original reactant molecule.
        fragments2: A list of fragments for product.
        mapping_list2: Atom indices of every heavy atoms in the original product molecule.
    '''
    nfrags1 = len(fragments1)
    nfrags2 = len(fragments2)
    if nfrags1 == 1 and nfrags2 == 1:
        matches = fragments1[0].GetSubstructMatches(fragments2[0], uniquify=False)
        result = []
        for match in matches:
            result.append([(mapping_list1[0][idx], mapping_list2[0][i]) for i, idx in enumerate(match)])
        return result
    
    if (nfrags1 == 1 and nfrags2 == 2) or (nfrags1 == 2 and nfrags2 == 1):
        one_fragment  = fragments1[0] if nfrags1 == 1 else fragments2[0] 
        two_fragments = fragments2    if nfrags1 == 1 else fragments1

        matches_part0 = one_fragment.GetSubstructMatches(two_fragments[0], uniquify=False)
        matches_part1 = one_fragment.GetSubstructMatches(two_fragments[1], uniquify=False)

        matches_part_dict0 = dict()
        for i, item in enumerate(matches_part0):
            sorted_item = tuple(sorted(item))
            if sorted_item in matches_part_dict0:
                matches_part_dict0[sorted_item].append(i)
            else:
                matches_part_dict0[sorted_item] = [i]
        
        matches_part_dict1 = dict()
        for i, item in enumerate(matches_part1):
            sorted_item = tuple(sorted(item))
            if sorted_item in matches_part_dict1:
                matches_part_dict1[sorted_item].append(i)
            else:
                matches_part_dict1[sorted_item] = [i]
        
        # 找出正确的索引对，它们需要组成整个分子
        natm = one_fragment.GetNumAtoms() 
        index = list(range(natm))
        good_key_pairs_list = []
        for key0 in matches_part_dict0.keys():
            for key1 in matches_part_dict1.keys():
                if sorted(key0 + key1) == index:
                    good_key_pairs_list.append((key0, key1))
        
        collect_matches = []
        if nfrags1 == 1:
            for key0, key1 in good_key_pairs_list:
                for i0 in matches_part_dict0[key0]:
                    match0 = [(mapping_list1[0][idx], mapping_list2[0][i]) for i, idx in enumerate(matches_part0[i0])]
                    for i1 in matches_part_dict1[key1]:
                        match1 = [(mapping_list1[0][idx], mapping_list2[1][i]) for i, idx in enumerate(matches_part1[i1])]
                        collect_matches.append(match0 + match1)
        else:
            for key0, key1 in good_key_pairs_list:
                for i0 in matches_part_dict0[key0]:
                    match0 = [(mapping_list1[0][i], mapping_list2[0][idx]) for i, idx in enumerate(matches_part0[i0])]
                    for i1 in matches_part_dict1[key1]:
                        match1 = [(mapping_list1[1][i], mapping_list2[0][idx]) for i, idx in enumerate(matches_part1[i1])]
                        collect_matches.append(match0 + match1)
        return collect_matches
    
    # 如果反应物和产物都各有两个分子片，此时合理的基元反应是基团转移反应 A1 + B1 <-> A2 + B2
    # 一个基团转移到另一个上时必定会导致A1和B1中一个原子变多，另一个变少
    if nfrags1 == 2 and nfrags2 == 2:
        collect_matches = []
        for a1,a2,b1,b2 in [(0,0,1,1), (0,1,1,0), (1,0,0,1), (1,1,0,0)]:
            if not fragments1[a1].GetSubstructMatch(fragments2[a2]): # a2 ⊆ a1
                continue
            if not fragments2[b2].GetSubstructMatch(fragments1[b1]): # b1 ⊆ b2
                continue 

            # a2 ⊆ a1
            matches_part_a1a2 = fragments1[a1].GetSubstructMatches(fragments2[a2], uniquify=False)
            matches_part_a1a2_dict = dict()
            for i, item in enumerate(matches_part_a1a2):
                sorted_item = tuple(sorted(item, reverse=True))
                if sorted_item in matches_part_a1a2_dict:
                    matches_part_a1a2_dict[sorted_item].append(i)
                else:
                    matches_part_a1a2_dict[sorted_item] = [i]
            
            # b1 ⊆ b2
            matches_part_b2b1 = fragments2[b2].GetSubstructMatches(fragments1[b1], uniquify=False)
            matches_part_b2b1_dict = dict()
            for i, item in enumerate(matches_part_b2b1):
                sorted_item = tuple(sorted(item, reverse=True))
                if sorted_item in matches_part_b2b1_dict:
                    matches_part_b2b1_dict[sorted_item].append(i)
                else:
                    matches_part_b2b1_dict[sorted_item] = [i]
            

            index = list(range(fragments2[b2].GetNumAtoms())) #片段 b2的原子数。A1-A2+B1 = B2
            for key_a1a2 in matches_part_a1a2_dict.keys():
                
                rw_mola = Chem.RWMol(fragments1[a1])
                for idx in key_a1a2:
                    rw_mola.RemoveAtom(idx) #现在的分子是从a1中去掉了匹配a2的部分
                
                mapping_list_remain = list(set(range(fragments1[a1].GetNumAtoms())).difference(set(key_a1a2)))
                
                match_remain = fragments2[b2].GetSubstructMatches(rw_mola, uniquify=False)
                match_remain_dict = dict()
                for i, item in enumerate(match_remain):
                    sorted_item = tuple(sorted(item, reverse=True))
                    if sorted_item in match_remain_dict:
                        match_remain_dict[sorted_item].append(i)
                    else:
                        match_remain_dict[sorted_item] = [i]

                good_key_pairs_list = []
                for key_remain in match_remain_dict.keys(): #验证A1 - A2 + B1 = B2
                    for key_b2b1 in matches_part_b2b1_dict.keys():
                        if sorted(key_remain + key_b2b1) == index:
                            good_key_pairs_list.append((key_remain, key_b2b1))
                
                for key_remain, key_b2b1 in good_key_pairs_list:
                    for i0 in matches_part_a1a2_dict[key_a1a2]:
                        match0 = [(mapping_list1[a1][idx], mapping_list2[a2][ii]) for ii, idx in enumerate(matches_part_a1a2[i0])]
                        for i1 in matches_part_b2b1_dict[key_b2b1]:
                            match1 = [(mapping_list1[b1][ii], mapping_list2[b2][idx]) for ii, idx in enumerate(matches_part_b2b1[i1])]
                            for i2 in match_remain_dict[key_remain]:
                                match2 = [(mapping_list1[a1][mapping_list_remain[ii]], mapping_list2[b2][idx]) for ii, idx in enumerate(match_remain[i2])]
                                collect_matches.append(match0 + match1 + match2)
        return collect_matches
    
    return []

        
def _match_index_by_permutation(atoms1:Atoms, 
                                atoms2:Atoms, 
                                only_permutations:bool = False) -> Tuple[list, Atoms, Atoms] | list:
    '''
    Helper function to match the atom indices by exhausting all permutations or just returns all possible permutations.
    Args:
        atoms1: `ase.Atoms` object for reactant.
        atoms2: `ase.Atoms` object for product.
        only_permutations: If `True`, just return all the permutations.
    '''
    count = Formula(atoms1.get_chemical_formula()).count()
    elements = list(count.keys())
 
    dict1 = {key:[] for key in elements}
    symbols1 = atoms1.get_chemical_symbols()
    for i, s in enumerate(symbols1):
        dict1[s].append(i)
    
    dict2 = {key:[] for key in elements}
    symbols2 = atoms2.get_chemical_symbols()
    for i, s in enumerate(symbols2):
        dict2[s].append(i)
    
    ori_index = []
    index_permutations = []
    for key in elements:
        ori_index.append(dict1[key])
        index_permutations.append(permutations(dict2[key]))
    permute_index = list(product(*index_permutations))

    ori_index = list(chain.from_iterable(ori_index))
    all_matches = []
    for item in permute_index:
        item_index = list(chain.from_iterable(item))
        all_matches.append([(i,j) for i,j in zip(ori_index, item_index)])

    if only_permutations:
        return all_matches

    pos2 = atoms2.positions - np.mean(atoms2.positions, axis=0) + np.mean(atoms1.positions, axis=0)
    dist_list = []
    for match in all_matches:
        index0 = []
        index1 = []
        for i,j in match:
            index0.append(i)
            index1.append(j)
        dist = np.sum((atoms1.positions[index0] - pos2[index1])**2)
        dist_list.append(dist)

    match = all_matches[np.argmin(dist_list)]
    match = sorted(match, key=lambda x:x[0])
    index1 = [a[1] for a in match]

    new_atoms = Atoms(numbers=atoms2.get_atomic_numbers()[index1], 
                      positions=atoms2.positions[index1])
    return (index1, atoms1, new_atoms)  


def match_atom_indices(atoms1:Atoms, 
                       atoms2:Atoms, 
                       n_permutation_all:int = 200,
                       n_permutation_heavy:int = 100,
                       skin:float = 0.1) -> Tuple[list, Atoms, Atoms]:
    '''
    Match the indices of atoms in reactant and product (`atoms1` and `atoms2`) to minimize the distance between them. This function operates in two modes: exhaustive search and structure matching. For small molecules, the exhaustive search mode is employed to avoid omitting potential optimal solutions during structure matching. For large molecules, structure matching mode is preferable to reduce computational workload. In structure matching mode, all H atoms are removed, molecular scaffolds are matched by finding the Maximum Common Substructure (MCS) or matching the substructure. Thus this function fails for high symmetry case. For example, in reaction CH2: + CH3-CH3 <-> CH3CH2CH3, the scaffold is C + CC <-> CCC, which will be taken as a coupling reaction. Another example is HO-CH2-CH <-> CH2-CH-OH, the scaffold is O(0)C(1)C(2) <-> C(1)C(2)O(0), C(1) will be mapped to C(2) because both reactant and product have CCO scaffold. To get correct result in structure matching mode, only use this function in the following cases:
    - Intramolecular reaction. In this case there is only one fragment in reactant and product. For example, CH2CH2CH2CH2 <-> cyclobutane.
    - Dissociation and coupling reaction. For example, CH3CH2 <-> CH3 + CH2
    - Atomic group migration between two molecules, in this case, there are two fragments in reactant and product. For example, CH3CHOH + CH2 <-> CH3CH + CH2OH
    - Diels-Alder reaction.
    Args:
        atoms1: `ase.Atoms` object for reactant, two molecular fragments at most.
        atoms2: `ase.Atoms` object for product, two molecular fragments at most.
        n_permutation_all: Threshold to enumerate the permutations of all the atoms.
        n_permutation_heavy: Threshold to enumerate the permutations of heavy atoms.
        skin: For atom `A` and `B`, if their distance satisfy $d(A-B) < skin*2 + r(A) + r(B)$, consider them to be bonded. Here r(A) and r(B) are the covalent radii.
    Returns:
        Returns the `ase.Atoms` object of reactant and product, in which `atoms1` is intacted but the indices in `atoms2` are reordered to minimize the distance.
    ''' 
    # 总原子数必须相同
    natoms = len(atoms1)
    if natoms != len(atoms2):
        raise ValueError("The number of atoms in `atoms1` and `atoms2` must be the same. ")
    
    # 每种元素的个数必须相同
    count1 = Formula(atoms1.get_chemical_formula()).count()
    count2 = Formula(atoms2.get_chemical_formula()).count()
    if sorted(list(count1.keys())) != sorted(list(count2.keys())):
        raise ValueError("The elements in `atoms1` and `atoms2` are not the same.")
    
    n_combs = 1
    n_combs_heavy = 1
    for key in count1.keys():
        if count1[key] != count2[key]:
            raise ValueError(f"The number of atoms {key} in `atoms1` and `atoms2` must be the same.")
        a = factorial(count1[key])
        n_combs *= a
        if key != 'H':
            n_combs_heavy *= a
    if n_combs < n_permutation_all or natoms < 6:
        return _match_index_by_permutation(atoms1, atoms2)
    
    # 如果是H解离反应，获取解离H的编号
    atoms1_fragments, atoms1_mapping_list = split_ase_atoms_fragments(atoms1, skin)
    atoms2_fragments, atoms2_mapping_list = split_ase_atoms_fragments(atoms2, skin)
    n_fragments1 = len(atoms1_fragments)
    n_fragments2 = len(atoms2_fragments)
    if n_fragments1 > 2 or n_fragments2 > 2:
        raise ValueError("There are more than 2 fragments in `atoms1` or `atoms2`. This is not a elementary step.")
    
    # 如果是H解离反应，这里假设有且只有一个H原子。其他平凡情形，例如H2 <-> H + H由上面的轮换解决
    H_index = None 
    if n_fragments1 == 2: 
        if len(atoms1_fragments[0]) == 1 and atoms1_fragments[0][0].symbol == 'H':
            H_index = -atoms1_mapping_list[0][0]
        elif len(atoms1_fragments[1]) == 1 and atoms1_fragments[1][0].symbol == 'H':
            H_index = -atoms1_mapping_list[1][0]
    elif n_fragments2 == 2:
        if len(atoms2_fragments[0]) == 1 and atoms2_fragments[0][0].symbol == 'H':
            H_index = atoms2_mapping_list[0][0]
        elif len(atoms2_fragments[1]) == 1 and atoms2_fragments[1][0].symbol == 'H':
            H_index = atoms2_mapping_list[1][0]
    
    # 1. 第一步，找出所有与重原子配位的H原子。在atoms对象中的原始编号
    numbers1 = atoms1.get_atomic_numbers()
    nl1 = NeighborList(natural_cutoffs(atoms1), self_interaction=False, bothways=True)
    nl1.update(atoms1)
    heavy_atoms_adj_H_dict1 = dict()
    for i in range(natoms):
        if numbers1[i] == 1:
            continue
        index, _ = nl1.get_neighbors(i)
        heavy_atoms_adj_H_dict1[i] = [int(j) for j in index if numbers1[j] == 1]
    
    numbers2 = atoms2.get_atomic_numbers()
    nl2 = NeighborList(natural_cutoffs(atoms2), self_interaction=False, bothways=True)
    nl2.update(atoms2)
    heavy_atoms_adj_H_dict2 = dict()
    for i in range(natoms):
        if numbers2[i] == 1:
            continue
        index, _ = nl2.get_neighbors(i)
        heavy_atoms_adj_H_dict2[i] = [int(j) for j in index if numbers2[j] == 1]
    
    # 2. 第二步，删除所有的H原子，求删除H原子前后重原子的编号对应
    flag1 = numbers1 != 1
    heavy_atoms_mapping_array1 = np.where(flag1)[0] # 这里获得的是第i个重原子在分子中的原始编号
    clean_atoms1 = Atoms(numbers=numbers1[flag1], positions=atoms1.positions[flag1])

    flag2 = numbers2 != 1
    heavy_atoms_mapping_array2 = np.where(flag2)[0]
    clean_atoms2 = Atoms(numbers=numbers2[flag2], positions=atoms2.positions[flag2])
    
    clean_mol1 = ase_atoms_to_mol(clean_atoms1)
    clean_mol2 = ase_atoms_to_mol(clean_atoms2)

    mapping_list1 = []
    heavy_fragments1 = Chem.GetMolFrags(mol=clean_mol1, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping_list1)
    mapping_list1 = [[int(heavy_atoms_mapping_array1[idx]) for idx in item] for item in mapping_list1]

    mapping_list2 = []
    heavy_fragments2 = Chem.GetMolFrags(mol=clean_mol2, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping_list2)
    mapping_list2 = [[int(heavy_atoms_mapping_array2[idx]) for idx in item] for item in mapping_list2]

    if n_combs_heavy < n_permutation_heavy or len(clean_atoms1) < 6:
        h1 = heavy_atoms_mapping_array1.tolist()
        h2 = heavy_atoms_mapping_array2.tolist()
        permutation_list = _match_index_by_permutation(clean_atoms1, clean_atoms2, True)
        all_matches = [[(h1[i], h2[j]) for i,j in item] for item in permutation_list]
    else:
        all_matches = _getall_matches(heavy_fragments1, mapping_list1, heavy_fragments2, mapping_list2)
        if len(all_matches) == 0:
            all_matches = _getall_matches_MCS(clean_mol1, heavy_atoms_mapping_array1, clean_mol2, heavy_atoms_mapping_array2)
 
    # 把分子2移动到分子1的中心
    pos2 = atoms2.get_positions()
    pos2 = pos2 - np.mean(pos2, axis=0) + np.mean(atoms1.positions, axis=0)
    collect_matches = []
    collect_diffH = []
    # 遍历每个骨架的匹配，选出符合要求的匹配。
    for match in all_matches:
        result = deepcopy(match)
        diff_H = sum([abs(len(heavy_atoms_adj_H_dict1[a]) - len(heavy_atoms_adj_H_dict2[b])) for a,b in match])
        collect_diffH.append(diff_H)
        H_has_been_used = False 
        wrong_match = False 
        if diff_H == 0 or (diff_H == 1 and (not H_index is None)):
            for a, b in match:
                index1 = deepcopy(heavy_atoms_adj_H_dict1[a])
                index2 = deepcopy(heavy_atoms_adj_H_dict2[b])

                # 如果两个原子的H配位数不相等
                if len(index1) != len(index2): 
                    if H_index is None:
                        # 没有额外的H，不是H解离反应，当前匹配是错误的
                        wrong_match = True 
                        break 
                    if H_has_been_used:
                        # 如果有额外的H，但是这个H已经被用掉了。在这个前提下仍然有不匹配的H配位数。则说明是错误匹配
                        wrong_match = True 
                        break 
                    # 有额外的H且 H没有被用掉，则现在使用它
                    H_has_been_used = True 
                    if H_index < 0:
                        index1.append(-H_index)
                    else:
                        index2.append(H_index)
                
                nH = len(index1)
                p2_list = list(permutations(index2))
                dist = [np.sum((atoms1.positions[index1] - pos2[list(p2)])**2) for p2 in p2_list]
                p2 = p2_list[np.argmin(dist)]
                result.extend([ (index1[i], p2[i]) for i in range(nH)])
            if wrong_match: #当前匹配是错误的匹配
                continue
            collect_matches.append(result)
    
    if len(collect_matches) == 0:
        # 这里来处理特殊情况，分子内的H转移反应。该反应的H配位数本来就是不匹配的，collect_matches只能是空
        # 但如果是H转移反应，H配位数的差别至多应该是2.
        for i, diff_H in enumerate(collect_diffH):
            if diff_H != 2:
                continue
            result = deepcopy(all_matches[i])
            mismatch_pair_list = []
            for a, b in all_matches[i]:
                index1 = deepcopy(heavy_atoms_adj_H_dict1[a])
                index2 = deepcopy(heavy_atoms_adj_H_dict2[b])
                n1 = len(index1)
                n2 = len(index2)
                if n1 != n2:
                    mismatch_pair_list.append((a,b))
                    continue 
                nH = n1
                p2_list = list(permutations(index2))
                dist = [np.sum((atoms1.positions[index1] - pos2[list(p2)])**2) for p2 in p2_list]
                p2 = p2_list[np.argmin(dist)]
                result.extend([ (index1[i], p2[i]) for i in range(nH)])
            
            if len(mismatch_pair_list) != 2:
                continue
            # 必须是两个原子各有一个不匹配的H
            # 把反应前后这个两个原子上的H索引相加
            index1 = heavy_atoms_adj_H_dict1[mismatch_pair_list[0][0]] + heavy_atoms_adj_H_dict1[mismatch_pair_list[1][0]]
            index2 = heavy_atoms_adj_H_dict2[mismatch_pair_list[0][1]] + heavy_atoms_adj_H_dict2[mismatch_pair_list[1][1]]
            if len(index1) != len(index2):
                continue
            nH = len(index1)
            p2_list = list(permutations(index2))
            dist = [np.sum((atoms1.positions[index1] - pos2[list(p2)])**2) for p2 in p2_list]
            p2 = p2_list[np.argmin(dist)]
            result.extend([ (index1[i], p2[i]) for i in range(nH)])
            collect_matches.append(result)
    
    if len(collect_matches) == 0:
        raise RuntimeError("This reaction may be not an elementary reaction. No reasonable atom mapping found.")
   
    
    dist = []
    for match in collect_matches:
        ordered_match = sorted(match, key=lambda x:x[0])
        order = [item[1] for item in ordered_match]
        dist.append(np.sum((atoms1.positions - atoms2.positions[order])**2))
    
    ii = np.argmin(dist)
    ordered_match = sorted(collect_matches[ii], key=lambda x:x[0])
    order = [item[1] for item in ordered_match]

 
    symbols = [atoms2.symbols[i] for i in order]
    new_atoms2 = Atoms(symbols=symbols, positions = atoms2.positions[order])
    return (order, atoms1, new_atoms2)

