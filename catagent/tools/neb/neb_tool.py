from catagent.libs.nebkit.structure.bind_to_surface import bind_ini_fin_to_surface, bind_single_ads_to_surface
from catagent.libs.nebkit.nebpath.surface_reactions import build_surface_neb_path
from catagent.tools.core import ToolOutput
from ase.io import read, write 
from typing import List 
import os 
from langchain_core.tools import tool


@tool
def sampling_ini_fin_states_on_surface(ini_fragments:List[str],
                                       fin_fragments:List[str],
                                       surface:str,
                                       save_path:str) -> dict:
    '''
    Build the adsorption configurations of initial and final states for NEB calculation. Automatically handle the matching of atomic indices, alignment of initial and final atomic positions, and traversal of surface adsorption sites. Generate three files: `ini_states.extxyz` and `fin_states.extxyz` containing the initial and final configurations, `adsorbate_atomic_indices.dat` containing the atomic indices of adsorbate.

    Use this tool if only a few NEB calculations are needed. If you want to calculate a chemical system, use reaction system tools to generate the reaction network and build adsorption configurations.

    Args:
        ini_fragments: Absolute pathes to the molecular fragment structures (`*.xyz` format) of initial state.
        fin_fragments: Absolute pathes to the molecular fragments (`*.xyz` format) of final state. 
        surface: Absolute path to surface structure.
        save_path: Absolute path to save output files.
    ''' 

    pathes = ini_fragments + fin_fragments
    pathes.append(surface)
    for p in pathes:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Error: {p} does not exist.")
    if not os.path.isdir(save_path):
        raise NotADirectoryError(f"Error: {save_path} is not a directory.")

    ini_atoms = [read(item) for item in ini_fragments]
    fin_atoms = [read(item) for item in fin_fragments]
    surf_atoms = read(surface)
    ini, fin = bind_ini_fin_to_surface(ini_atoms, fin_atoms, surf_atoms)

    path_to_ini = os.path.join(save_path, "ini_states.extxyz")
    write(path_to_ini, ini)

    path_to_fin = os.path.join(save_path, "fin_states.extxyz")
    write(path_to_fin, fin)

    path_to_indices = os.path.join(save_path, "adsorbate_atomic_indices.dat")
    with open(path_to_indices, mode="w") as f:
        nsurf_atoms = len(surf_atoms)
        ntot = len(ini[0])
        for i in range(nsurf_atoms, ntot):
            f.write(str(i)+'\n')

    return ToolOutput(content=f"Initial and final states structures are written to \
                      {path_to_ini} and {path_to_fin}. Indices of adsorbate are written to {path_to_indices}.", 
                      outfile_list=[path_to_ini, path_to_fin, path_to_indices]).model_dump(mode='python')

@tool 
def build_initial_nebpath_on_surface(save_path: str,
                                      ini_state_path: str,
                                      fin_state_path: str,
                                      adsorbate_indices_path: str,
                                      image_spacing: float = 0.5,
                                      n_images: int = None):
    '''
    Make interpolation between relaxed IS and FS to get an initial NEB path. Automatically handle some structure problems, such as molecular dissociation in vacuum and surface atomic movement. Generate a `nebpath.extxyz` file containing the initial neb path.

    Args:
        save_path: Absolute path to save the `nebpath.extxyz`.
        ini_state_path: Absolute path to initial state structure.
        fin_state_path: Absolute path to final state structure.
        adsorbate_indices_path: Absolute path to the data file which containing the atomic indices of adsorbate.
        image_spacing: Distance between two adjacent NEB images (in Angstroms). This argument is ignored if `n_images` is not None.
        n_images: The number of NEB images.
    '''
    if not os.path.isdir(save_path):
        raise NotADirectoryError(f"Error: {save_path} is not a directory.")
    for p in [ini_state_path, fin_state_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Error: {p} is not a file.")
        
    ini_state = read(ini_state_path)
    fin_state = read(fin_state_path)
    ads_index = []
    with open(adsorbate_indices_path, mode="r") as f:
        lines = f.readlines()
        ads_index = [int(line) for line in lines]
    result = build_surface_neb_path(ini_state, fin_state, ads_index, image_spacing, n_images)
    filepath = os.path.join(save_path, "nebpath.extxyz")
    write(filepath, result)
    return ToolOutput(content=f"Initial neb path is written to {filepath}.", outfile_list=[filepath]).model_dump(mode='python')


@tool
def sampling_single_ads_configs_on_surface(source_surface_structure:str,
                                            source_adsorbate_structure:str,
                                            save_path:str) -> dict:
    '''
    Bind an single adsorbate to the surface, sampling all possible binding sites according to the chemical environment. Generate an `single_ads_configs.extxyz` file which contains all the configurations and an `adsorbate_atomic_indices.dat` containing the atomic indices of adsorbate. (Do not use this function to build NEB initial or final structures, because it does not guarantee the correspondence of atom indices.)

    Args:
        source_surface_structure: Absolute path to the structure of surface.
        source_adsorbate_structure: Absolute path to the structure of adsorbate.
        save_path: Absolute path to save output files.
    '''
    for a in [source_surface_structure, source_adsorbate_structure]:
        if not os.path.isfile(a):
            raise FileNotFoundError(f"{a} is not a file.")
    if not os.path.isdir(save_path):
        raise NotADirectoryError(f"{save_path} is not a directory.")

    surface   = read(source_surface_structure)
    adsorbate = read(source_adsorbate_structure)
    list_of_configs = bind_single_ads_to_surface(adsorbate, surface)

    filename = os.path.join(save_path, "single_ads_configs.extxyz")
    write(filename, list_of_configs, format="extxyz")

    indices_filename = os.path.join(save_path, "adsorbate_atomic_indices.dat")
    nsurf_atoms = len(surface)
    nads_atoms = len(adsorbate)
    ntot = nsurf_atoms + nads_atoms
    with open(indices_filename, mode="w") as f:
        content = '\n'.join([str(i) for i in range(nsurf_atoms, ntot)])
        f.write(content)

    return ToolOutput(content=f"Adsorption configurations are written to {filename}. \
                      Atomic indices of adsorbate are written to {indices_filename}", 
                      outfile_list=[filename, indices_filename]).model_dump(mode='python')
