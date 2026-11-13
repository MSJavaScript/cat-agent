from catagent.libs.nebkit.network.system import ReactionSystem
from typing import Dict, List, Tuple, Literal
from ase.io import read, write 
import os, pickle, json
from langchain.tools import tool, ToolRuntime


class ReactionSystemToolSets:
    def __init__(self):
        self.reactions_systems:Dict[str, Dict[str, ReactionSystem]] = dict() 

    def create(self):
        this = self

        @tool 
        def dump(runtime:ToolRuntime,
                 system_name: str, 
                 file_path:str):
            '''
            Dump a reaction system object to a pkl file.
            Args:
                system_name: Name of the reaction system.
                file_path: Absolute path to a pkl file.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            with open(file_path, mode="wb") as f:
                obj = this.reactions_systems[thread_id]
                pickle.dump(obj)
            return f"Reaction system {system_name} is dumped to {file_path}."

        @tool 
        def load_from_file(system_name: str, file_path:str, runtime:ToolRuntime):
            '''
            Load a reaction system object from the pickle file and assign a name to it.
            Args:
                system_name: Name of the reaction system.
                file_path: Absolute path to the pickle file which contains a reaction system object.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            with open(file_path, mode="rb") as f:
                obj = pickle.load(f)
                if not thread_id in this.reactions_systems:
                    this.reactions_systems[thread_id] = dict()
                this.reactions_systems[thread_id][system_name] = obj 
            return f"ReactionSystem object is loaded from {file_path}, with key {system_name}." 

        @tool
        def has_reaction_system(system_name: str, runtime:ToolRuntime)->bool:
            '''
            Cheack if a reaction system exist.
            Args:
                system_name: Name of the reaction system.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if not thread_id in this.reactions_systems:
                return False 
            if not system_name in this.reactions_systems[thread_id]:
                return False 
            return True

        @tool
        def delete_reaction_system(system_name: str, runtime:ToolRuntime) -> str:
            '''
            Delete a reaction system.
            Args:
                system_name: Name of the reaction system.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if not thread_id in this.reactions_systems:
                return "No reaction system exists in the current thread."
            if not system_name in this.reactions_systems[thread_id]:
                return f"{system_name} does not exist in current thread."
            this.reactions_systems[thread_id].pop(system_name)
            return f"{system_name} is deleted from current thread."


        @tool 
        def create_reaction_system(
            runtime:ToolRuntime,
            system_name: str,
            reactants_mol_pathes:List[str],
            products_mol_pathes:List[str],
            json_log_path: str,
            visualization_dir: str):
            '''
            Create a `ReactionSystem` object. A molecule should only appear one time. For example, for reaction "N2 + 3*H2 <-> 2*NH3", just call this tool by `reactants_mol_pathes=['/path/to/N2.xyz', '/path/to/H2']` and `products_mol_pathes = ['/path/to/NH3']`. 
            Args:
                system_name: A short name of the reaction system, e.g. 'CO2,CH4->CO,H2,H2O'.
                reactants_mol_pathes: Path to reactant molecules files (usually in `.xyz` format).
                products_mol_pathes: Path to product molecules files.
                json_log_path: Path to the json file which logs the state of `ReactionSystem` object.
                visualization_dir: The dir path to save visualization pictures of intermediate species and reactions.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if thread_id in this.reactions_systems and system_name in this.reactions_systems[thread_id]:
                raise RuntimeError(f"Reaction system {system_name} already exists.")

            reactants = []
            for item in reactants_mol_pathes:
                if not os.path.isfile(item):
                    raise FileNotFoundError(f"Error: {item} is not a file.")
                try:
                    reactants.append(read(item))
                except Exception as e:
                    raise RuntimeError(f"Error: {item} can not be read by ase. {e}")
            
            products = []
            for item in products_mol_pathes:
                if not os.path.isfile(item):
                    raise FileNotFoundError(f"Error: {item} is not a file. ")
                try:
                    products.append(read(item))
                except Exception as e:
                    raise RuntimeError(f"Error: {item} can not be read by ase. {e}")

            if not thread_id in this.reactions_systems:
                this.reactions_systems[thread_id] = dict()

            rs = ReactionSystem(reactants, products)
            rs.json_log_path = json_log_path
            rs.visualization_dir = visualization_dir

            this.reactions_systems[thread_id][system_name] = rs 
            msg = f"Reaction system {system_name} is created. To generate reactions, call `generate_elementary_steps`."           
            return msg

        @tool 
        def render(runtime:ToolRuntime, system_name: str, backend:Literal["ase", "rdkit"] = "rdkit"):
            '''
            Render the intermediates and reactions to pictures.
            Args:
                system_name: Name of the reaction system.
                backend: The backend used to render the reaction system.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]

            if not os.path.isdir(rs.visualization_dir):
                os.makedirs(rs.visualization_dir, exist_ok=True)

            result = os.listdir(rs.visualization_dir)
            for f in result:
                os.remove(os.path.join(rs.visualization_dir, f))

            msg = rs.render_system(rs.visualization_dir, backend=backend)
            return msg

        @tool 
        def generate_elementary_steps(
            runtime:ToolRuntime, 
            system_name: str,
            ntimes:int = -1,
            max_natoms_tot:int = -1, 
            min_natoms_tot:int = 0, 
            max_natoms:Dict[str, int] = None, 
            min_natoms:Dict[str, int] = None, 
            forbidden_pattern: List[Tuple[str, str]] = None):
            '''
            Generate elementary steps (reactions), `create_reaction_system` should be called firstly to create a `ReactionSystem` object. This is a human-in-the-loop (HITL) tool.
            Args:
                system_name: Name of the reaction system, the corresponding `ReactionSystem` object should be created in advance.
                ntimes: `ntimes = -1` (default) use BFS to find all possible elementary steps. `ntimes > 0` sample the elementary steps n times using DFS.
                max_natoms_tot: Max number of atoms in the intermediates. `-1` means no limitation.
                min_natoms_tot: Min number of atoms in the intermediates.
                max_natoms: A `dict` used to limit the max number of atoms of each element in the intermediates. If omitted, a default value is calculated from the final products.
                min_natoms: A `dict` used to limit the min number of atoms of each element in the intermediates.
                forbidden_pattern: A list of forbidden substructures represented by molecular graph, e.g., `('H,O,C,O,H','0-1,1-2,2-3,3-4')` represents geminal diol. If omitted, a list of default patterns are generated.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                  (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]
            rs.generate_elementary_steps(ntimes=ntimes,
                                         max_natoms_tot=max_natoms_tot,
                                         min_natoms_tot=min_natoms_tot,
                                         max_natoms=max_natoms,
                                         min_natoms=min_natoms,
                                         forbidden_pattern=forbidden_pattern)
            
            msg = rs.dump_info(rs.json_log_path)
            return f"{len(rs.reactions_list)} reactions are generated. Call `render` to visualize the reactions. {msg}"


        @tool
        def delete_species(runtime:ToolRuntime,
                           system_name: str, 
                           species_indices: List[int] = None):
            '''
            Delete intermediate speices from the system. Related reactions will also be deleted. This is a human-in-the-loop (HITL) tool.
            Args:
                system_name: Name of the reaction system.
                species_indices: Indices of intermediate speices.
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]
            for i in species_indices:
                rs.delete_species(i)
            msg = rs.dump_info(rs.json_log_path)
            return f"Intermediates {species_indices} are deleted. {msg}"

        @tool
        def delete_reactions(runtime:ToolRuntime,
                             system_name: str, 
                             reaction_indices: List[int] = None):
            '''
            Delete reactions from the system. Related intermediate speices will also be deleted. This is a human-in-the-loop (HITL) tool.
            Args:
                system_name: Name of the reaction system.
                reaction_indices: Indices of reactions. 
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]
            index = sorted(reaction_indices, reverse=True)
            for i in index:
                rs.delete_reaction(i)
            msg = rs.dump_info(rs.json_log_path)
            return f"Reactions {reaction_indices} are deleted. {msg}"

        @tool 
        def merge_reactions(runtime:ToolRuntime,
                            system_name: str, 
                            reaction_index_pairs: List[Tuple[int, int]] = None):
            '''
            Merge reactions in the system. This is a human-in-the-loop (HITL) tool.
            Args:
                system_name: Name of the reaction system.
                reaction_index_pairs: A list of tuples, each tuple contains the indices of reactions need to be merged. 
            ''' 
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]

            for a, b in reaction_index_pairs:
                rs.merge_reactions(a, b)
            
            msg = rs.dump_info(rs.json_log_path)
            return f"Reaction pairs {reaction_index_pairs} are merged. Call `delete_reactions` to delete redundant reactions. {msg}"

        @tool 
        def bind_to_surface(runtime:ToolRuntime,
                            system_name: str,
                            surface_structures_path:List[str],
                            cwd:str = None):
            '''
            Build initial and final states (in POSCAR format) of elementary reactions on the given surfaces. The result will be put in dirs `surface_<surface_index>/reaction_<reaction_index>` and will be named by `POSCAR_<state>_<conf_index>`. Here `<state>` is `ini` for `fin`, `<conf_index>` is the index of configuration.
            Args:
                system_name: Name of the reaction system.
                surface_structures_path: Pathes to surface structures, should be readable to ase.
                cwd: Current working directory. 
            '''
            thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]

            old_wd = os.getcwd()
            if cwd:
                os.chdir(cwd)
            
            for i, p in enumerate(surface_structures_path):
                info = dict()
                info["surface_structure"] = p 
                surface_atoms = read(p)
                result = rs.bind_to_surfaces([surface_atoms])
                os.makedirs(f"surface_{i}", exist_ok=True)
                intermediate_index_dict, reactions_result_list = result[0]
                info["n_reactions"] = len(reactions_result_list)
                reactions_info = []
                for j, reac in enumerate(reactions_result_list):
                    os.makedirs(f"surface_{i}/reaction_{j}", exist_ok=True)
                    reactions_info.append({
                        "reaction_string": reac["name"],
                        "ads_atom_indices": reac["ads_atoms_index"]
                    })
                    ini_list = reac["ini"]
                    fin_list = reac["fin"]
                    for k in range(len(ini_list)):
                        write(f"surface_{i}/reaction_{j}/POSCAR_ini_{k}", ini_list[k])
                        write(f"surface_{i}/reaction_{j}/POSCAR_fin_{k}", fin_list[k])
                info["reactions_info"] = reactions_info
                with open(f"surface_{i}/binding_info.json", mode="w") as f:
                    json.dump(info, indent=4)
            if cwd:
                os.chdir(old_wd)

            return f"The POSCARs initial and final states are written to `surface_*` dirs in {cwd if cwd else old_wd}. `surface_*/binding_info.json` contains the smiles string of each reaction and the indices of adsorbate atoms."

            
        return [dump, load_from_file, has_reaction_system, delete_reaction_system, 
                create_reaction_system, render, generate_elementary_steps, 
                delete_species, merge_reactions, delete_reactions, bind_to_surface]
    
