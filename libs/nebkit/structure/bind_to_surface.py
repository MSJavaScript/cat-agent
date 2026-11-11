from ase.neighborlist import NeighborList, natural_cutoffs 
from ase import Atoms 
from ase.data import covalent_radii
from ase.optimize import MDMin

from typing import List, Tuple 

from itertools import product

from scipy.spatial import Delaunay 
import numpy as np 
from rdkit import Chem 
from random import sample 

from libs.nebkit.calculator.atom_attractor import AtomAttractor
from libs.nebkit.calculator.match_reactants_and_products import RigidMDMin, AlignIniFinAtoms
from libs.nebkit.tools.structure import group_surface_layers
from libs.nebkit.tools.utils import get_adsorbate_binding_atom_index, ase_atoms_to_mol
from libs.nebkit.tools.matching import match_atom_indices
from libs.nebkit.network.helper import _combine_atoms


def classify_surface_atoms(atoms:Atoms, indices:np.ndarray|list):
    '''
    计算一个原子的广义配位数GCN，gcn用于原子位点的快速分类。
    '''
    # 该体系中有几种原子？
    
    symbols = atoms.get_chemical_symbols()
    atom_types = list(set(symbols))
    n_types = len(atom_types)

    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    natoms = len(atoms)
    neighbor_info = []
    
    for i in range(natoms):
        idx, offset = nl.get_neighbors(i)
        count_dict = {key:0 for key in atom_types}
        for j in idx:
            count_dict[symbols[j]] += 1
        neighbor_info.append({ "idx": idx,  "des": np.array([count_dict[key] for key in atom_types]) })

    
    collect_result = []
    for i in indices:
        des = np.zeros((n_types, ))
        for j in neighbor_info[i]["idx"]:
            des += neighbor_info[j]["des"]
        collect_result.append(des)

    sites_groups = [] # 根据gcn对位点进行分组
    for idx, des in zip(indices, collect_result):
        for group in sites_groups:
            if np.all(np.isclose(des, collect_result[indices.index(group[0])])):
                group.append(idx)
                break 
        else:
            sites_groups.append([idx])
    return sites_groups


def get_outmost_atoms_for_surface(atoms:Atoms)->Tuple[List[int], np.ndarray]:
    '''
    使用三角形剖分找表面最表层的原子。这里假设表面真空层沿z方向，在z方向按照坐标对原子层分组。从上到下依次把原子层加进来，对(x,y)坐标点做三角形剖分。直到三角形的直径小于最小原子距离的1.5倍。使用这种方式可以找到带有缺陷的结合位点，例如FCC(111)去掉表层一些原子，此时下一层原子暴漏到表面。
    '''
    dist_mat = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dist_mat, np.inf)
    threshold = np.min(dist_mat) * 1.5
    cell = atoms.get_cell()

    positions_ori = atoms.get_positions()
    collect_positions = [positions_ori.copy()]
    repeats = list(product([-1, 0, 1], repeat=2))
    repeats.remove((0, 0))
    for repeat in repeats:
        positions_shifted = positions_ori.copy()
        positions_shifted += repeat[0]*cell[0] + repeat[1]*cell[1]
        collect_positions.append(positions_shifted)
    positions = np.vstack(collect_positions)
    tiled_atoms = Atoms(numbers = np.zeros((positions.shape[0], ), dtype=int)+1,
                        positions = positions,
                        cell = cell)
    indices = group_surface_layers(tiled_atoms)
    collect_indices = []
    ngroups = len(indices)
    for i in range(ngroups - 1, -1, -1):
        if i == ngroups - 1:
            collect_indices.extend(indices[i].tolist()) 
            tri = Delaunay(positions[collect_indices, :2])
        else:
            idx_of_simplex = tri.find_simplex(positions[indices[i], :2])
            collect_indices.extend([ int(indices[i][j]) for j in range(len(indices[i])) \
                                    if idx_of_simplex[j] == -1 or (not simplex_is_small_enough[idx_of_simplex[j]] ) ])
            tri = Delaunay(positions[collect_indices, :2])
        # 求每个小三角形三个点之间的距离
        simplices = tri.simplices
        indices_array = np.array(collect_indices, dtype=int)
        p0 = positions[indices_array[simplices[:, 0]], :2]
        p1 = positions[indices_array[simplices[:, 1]], :2]
        p2 = positions[indices_array[simplices[:, 2]], :2]
        d01 = np.sqrt(np.sum((p0 - p1)**2, axis=1))
        d02 = np.sqrt(np.sum((p0 - p2)**2, axis=1))
        d12 = np.sqrt(np.sum((p1 - p2)**2, axis=1))
        dist = np.vstack((d01, d02, d12))
        max_dist = np.max(dist, axis=0)
        simplex_is_small_enough = max_dist < threshold
        if np.all(simplex_is_small_enough):
            break 
    natoms = len(atoms)
    collect_indices = [i for i in collect_indices if i < natoms]

    np.fill_diagonal(dist_mat, 0)
    return collect_indices, dist_mat

def get_gcn(atoms:Atoms, indices:np.ndarray|list):
    '''
    计算一个原子的广义配位数GCN，gcn用于原子位点的快速分类。
    '''
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    natoms = len(atoms)
    neighbor_list = []
    cn_max = 0
    for i in range(natoms):
        idx, offset = nl.get_neighbors(i)
        neighbor_list.append(idx)
        n_neighbor = len(idx)
        if n_neighbor > cn_max:
            cn_max = n_neighbor
    collect_result = []
    for i in indices:
        gcn = 0
        for j in neighbor_list[i]:
            gcn += len(neighbor_list[j])/float(cn_max)
        collect_result.append(gcn)
    return collect_result 



def _bind_single_ads_to_surface(ads:Atoms,
                                ads_radius:float,
                                ads_center: np.ndarray,
                                ads_binding_index:List[int],
                                surface:Atoms,
                                sites_groups:List[List[int]]):
    
    # 是否是单个原子？如果是单个原子生成两个构型
    single_atom = True if len(ads) == 1 else False 
    # 表面原子个数
    nsurf_atom = len(surface)
    # 表面原子的最大半径
    max_radius = np.max([ covalent_radii[a] for a in surface.get_atomic_numbers() ])
    result = []

    for group in sites_groups:
        bind_index = sample(group, 2 if single_atom else 1)
        for b_i in bind_index:
            new_center = np.array([surface.positions[b_i, 0],
                                surface.positions[b_i, 1],
                                surface.positions[b_i, 2] + ads_radius + 2.0])
            new_ads = ads.copy()
            #减去旧的中心换成新的中心
            new_ads.set_positions(ads.positions - ads_center + new_center) 
            combined = surface + new_ads 
            # 把吸附物吸引到指定的位点
            combined.calc = AtomAttractor(nsurf_atoms=nsurf_atom, atom_group_pairs=[(ads_binding_index, [b_i])])
            opt = MDMin(combined, logfile='tmp.log', trajectory='tmp.traj')
            opt.run(fmax=0.2, steps=50)
            # 在x和y方向加上一些扰动
            combined.positions[nsurf_atom:, 0:2] += np.random.random((2, )) * max_radius - 0.5*max_radius
            result.append(combined)
    
    return result


# 如果表面和吸附物数量较少，可以直接调用该函数
def bind_single_ads_to_surface(ads:Atoms,
                               surface:Atoms,
                               nsample:int = -1) -> List[Atoms]:
    '''
    Bind an adsorbate to the surface, sampling all possible binding sites.
    Args:
        ads: Adsorbate `ase.Atoms` object.
        surface: Surface `ase.Atoms` object.
        nsample: The number of max samples.
    ''' 

    # 获取吸附物的吸附原子
    index = get_adsorbate_binding_atom_index(ads)
    ads_radius = np.max(ads.get_all_distances(mic=True))*0.5 \
        + np.max([ covalent_radii[a] for a in ads.get_atomic_numbers() ])
    ads_center = np.mean(ads.positions, axis=0)
    # 如果是物理吸附，直接取表面的中心
    if index is None:
        surf_center = np.mean(surface.positions, axis=0)
        new_center = np.array([surf_center[0], 
                               surf_center[1],  
                               np.max(surface.positions[:, 2]) + ads_radius + 1.0])
        new_ads = ads.copy()
        new_ads.set_positions(ads.positions - ads_center + new_center)
        return [ surface + new_ads ]

    
    
    surface_sites, dist_mat = get_outmost_atoms_for_surface(surface)
    sites_groups = classify_surface_atoms(surface, surface_sites)
    result = _bind_single_ads_to_surface(ads = ads,
                                         ads_radius = ads_radius,
                                         ads_center = ads_center,
                                         ads_binding_index = index,
                                         surface = surface,
                                         sites_groups = sites_groups)
    if nsample > 0:
        result = result[0:nsample]
    return result
    


def _bind_ini_fin_to_surface(total_atoms:Atoms,
                             total_atoms_radius: float,
                             total_atoms_center: np.ndarray,
                             total_binding_index: List[List[int]],
                             mapping_index: List[int],
                             surface: Atoms,
                             dist_mat:np.ndarray,
                             sites_groups: List[List[int]],
                             is_symmetric:bool = False):
    '''
    Args:
        total_atoms: 反应物 + 产物的ase.Atoms对象。
        total_atoms_radius: 反应物 + 产物的Atoms对象的直径。
        total_atoms_center: 反应物 + 产物的Atoms对象的中心。
        total_binding_index: 反应物+产物中结合原子的索引。
        mapping_index: 反应物到产物原子序号的对应。
        surface: 表面的ase.Atoms对象。
        dist_mat: 原子之间的距离矩阵。
        sites_groups: 表面位点的分组
    '''
    # 如果初末态都只有一个片段，则只需要遍历依次位点
    nsurf_atoms = len(surface)
    natoms = int(len(total_atoms)/2)
    # 表面原子的最大半径
    max_radius = np.max([ covalent_radii[a] for a in surface.get_atomic_numbers() ])

    ini_result = []
    fin_result = []
    ads_index = []
    if len(total_binding_index) == 1:
        ii = 0
        for group in sites_groups:
            ads_index.append(ii)
            ii += 1
            sites = sample(group, 1)
            # 把total_atoms 放在位点上方
            new_total_atoms:Atoms = total_atoms.copy()
            new_center = np.array([surface.positions[sites[0], 0],
                                   surface.positions[sites[0], 1],
                                   surface.positions[sites[0], 2] + total_atoms_radius])
            new_total_atoms.set_positions(total_atoms.positions - total_atoms_center + new_center)
            surf_and_total_atoms:Atoms = surface + new_total_atoms
            surf_and_total_atoms.calc = AtomAttractor(nsurf_atoms, [(total_binding_index, sites)])
            opt = MDMin(surf_and_total_atoms, logfile='tmp.log', trajectory='tmp.traj')
            opt.run(fmax=0.2, steps=50)
        
            ini_state_index = np.arange(nsurf_atoms + natoms)
            fin_state_index = np.concatenate((np.arange(nsurf_atoms), np.array(mapping_index) + nsurf_atoms + natoms))
            ini_result.append(Atoms(cell = surf_and_total_atoms.get_cell(),
                                    numbers = surf_and_total_atoms.numbers[ini_state_index],
                                    positions = surf_and_total_atoms.positions[ini_state_index],
                                    pbc = [True, True, True]))
            fin_result.append(Atoms(cell = surf_and_total_atoms.get_cell(),
                                    numbers = surf_and_total_atoms.numbers[fin_state_index],
                                    positions = surf_and_total_atoms.positions[fin_state_index],
                                    pbc = [True, True, True]))
    else: # 有两个片段
        # 获取所有的合理位点组合以及新的中心
        ngroup = len(sites_groups)
        ii = 0
        for i in range(ngroup):
            for j in range(i, ngroup):
                sites_dist = [(a,b, dist_mat[a,b]) for a,b in product(sites_groups[i], sites_groups[j]) if dist_mat[a,b] > 2.0]
                sites_dist.sort(key = lambda x: x[2])
                a, b, dist = sites_dist[0]
                if dist > total_atoms_radius + max_radius*4:
                    continue
                vec = surface.get_distance(a, b, mic=True, vector=True)
                new_center = 0.5 * vec + surface.positions[a]
                new_center[2] += total_atoms_radius + 3.0

                new_total_atoms:Atoms = total_atoms.copy()

                # 先减去自身的中心方便执行旋转操作
                new_total_atoms.set_positions(total_atoms.positions - total_atoms_center)
                new_total_atoms.rotate(np.random.random()*90-45, np.random.random((3, ))-0.5)
                new_total_atoms.positions += new_center # 然后设置新的中心
                
                surf_and_total_atoms:Atoms = surface + new_total_atoms
                surf_and_total_atoms_2 = surf_and_total_atoms.copy()

                # 初态和末态的索引
                ini_state_index = np.arange(nsurf_atoms + natoms)
                fin_state_index = np.concatenate((np.arange(nsurf_atoms), np.array(mapping_index) + nsurf_atoms + natoms))

                surf_and_total_atoms.calc = AtomAttractor(nsurf_atoms, 
                                                          [(total_binding_index[0], [a]), (total_binding_index[1], [b])])
                
                opt = RigidMDMin(surf_and_total_atoms, 
                                 [np.arange(nsurf_atoms, len(surf_and_total_atoms))],
                                 logfile='tmp.log', trajectory='tmp.traj')
                # opt = MDMin(surf_and_total_atoms, logfile='tmp.log', trajectory='tmp.traj')
                opt.run(fmax=0.2, steps=100)
                ini_result.append(Atoms(cell = surf_and_total_atoms.get_cell(),
                                    numbers = surf_and_total_atoms.numbers[ini_state_index],
                                    positions = surf_and_total_atoms.positions[ini_state_index],
                                    pbc = [True, True, True]))
                
                fin_result.append(Atoms(cell = surf_and_total_atoms.get_cell(),
                                    numbers = surf_and_total_atoms.numbers[fin_state_index],
                                    positions = surf_and_total_atoms.positions[fin_state_index],
                                    pbc = [True, True, True]))
                
                if i == j:
                    ads_index.append(ii)
                ii += 1
                if i != j and (not is_symmetric): # 只有当两个结合位点不同时，考虑两个物种的互换
                    ii += 1
                    surf_and_total_atoms_2.calc = AtomAttractor(nsurf_atoms, 
                                                                [(total_binding_index[0], [b]), (total_binding_index[1], [a])])
                    opt = RigidMDMin(surf_and_total_atoms_2,
                                     [np.arange(nsurf_atoms, len(surf_and_total_atoms_2))],
                                     logfile='tmp2.log', trajectory='tmp2.traj')
                    
                    #opt = MDMin(surf_and_total_atoms_2, logfile='tmp2.log', trajectory='tmp2.traj')
                    opt.run(fmax=0.2, steps=100)
                    ini_result.append(Atoms(cell = surf_and_total_atoms_2.get_cell(),
                                        numbers = surf_and_total_atoms_2.numbers[ini_state_index],
                                        positions = surf_and_total_atoms_2.positions[ini_state_index],
                                        pbc = [True, True, True]))
                    
                    fin_result.append(Atoms(cell = surf_and_total_atoms_2.get_cell(),
                                        numbers = surf_and_total_atoms_2.numbers[fin_state_index],
                                        positions = surf_and_total_atoms_2.positions[fin_state_index],
                                        pbc = [True, True, True]))
            
    return ini_result, fin_result, ads_index

# 如果NEB的数据较少，可以调用该函数
def bind_ini_fin_to_surface(ini:List[Atoms],
                            fin:List[Atoms],
                            surface:Atoms,
                            nsample:int = -1) -> Tuple[List[Atoms], List[Atoms]]:
    '''
    Sample the co-adsorption configurations of ini and fin states of NEB calculation.
    '''
    ini_smiles_list = [Chem.MolToSmiles(ase_atoms_to_mol(item)) for item in ini]
    ini_combined = _combine_atoms(ini)
    ini_combined.set_positions(ini_combined.positions - np.mean(ini_combined.positions, axis=0))

    fin_smiles_list = [Chem.MolToSmiles(ase_atoms_to_mol(item)) for item in fin]
    fin_combined = _combine_atoms(fin)
    fin_combined.set_positions(fin_combined.positions - np.mean(fin_combined.positions, axis=0))

    # 匹配初末态的原子序号
    mapping_index, _, _ = match_atom_indices(ini_combined, fin_combined)
    start = 0
    reactants_index = []
    products_index = []
    for atoms in ini:
        reactants_index.append(start + np.arange(len(atoms)))
        start += len(atoms)
    for atoms in fin:
        products_index.append(start + np.arange(len(atoms)))
        start += len(atoms)

    total_atoms = ini_combined + fin_combined
    total_atoms.calc = AlignIniFinAtoms(reactants_index, products_index, mapping_index, total_atoms)
    opt = RigidMDMin(total_atoms, 
                        index_group=reactants_index + products_index,
                        logfile="RigidMDMin.log", 
                        trajectory="RigidMDMin.traj")
    opt.run(fmax=0.2, steps=50)

    total_atoms_radius = np.max(total_atoms.get_all_distances(mic=True))*0.5 \
        + np.max([ covalent_radii[a] for a in ini_combined.get_atomic_numbers() ])
    total_atoms_center = np.mean(total_atoms.positions, axis=0)


    surface_sites, dist_mat = get_outmost_atoms_for_surface(surface)
    sites_groups = classify_surface_atoms(surface, surface_sites)
    reactants_binding_index = []
    products_binding_index = []

    start = 0
    for frag in ini:
        bidx = get_adsorbate_binding_atom_index(frag)
        if not bidx is None:
            reactants_binding_index.append([int(bid+start) for bid in bidx])
        else:
            reactants_binding_index.append(None)
        start += len(frag)
    
    for frag in fin:
        bidx = get_adsorbate_binding_atom_index(frag)
        if not bidx is None:
            products_binding_index.append([int(bid+start) for bid in bidx])
        else:
            products_binding_index.append(None)
        start += len(frag)
         
    # 找结合的原子
    nfrag_ini = len(ini)
    nfrag_fin = len(fin)

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
    
    ini_result, fin_result, ads_index = _bind_ini_fin_to_surface(total_atoms=total_atoms,
                                                      total_atoms_radius=total_atoms_radius,
                                                      total_atoms_center=total_atoms_center,
                                                      total_binding_index=total_binding_index,
                                                      mapping_index=mapping_index,
                                                      surface=surface,
                                                      dist_mat=dist_mat,
                                                      sites_groups=sites_groups,
                                                      is_symmetric=is_symmetric)
    if nsample > 0:
        return (ini_result[0:nsample], fin_result[0:nsample])
    else:
        return (ini_result, fin_result)
