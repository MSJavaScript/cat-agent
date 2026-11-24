from nebkit import ReactionSystem
from typing import Dict, List, Tuple, Literal, Any
from ase.io import read, write 
import os, pickle, json
from langchain.tools import tool, ToolRuntime


class ReactionSystemToolSets:
    def __init__(self):
        self.reactions_systems:Dict[str, Dict[str, ReactionSystem]] = dict() 

    def create(self):
        this = self

        @tool 
        def dump(runtime:ToolRuntime[Any],
                 system_name: str, 
                 file_path:str):
            '''
            Dump a reaction system object to a pkl file.
            Args:
                system_name: Name of the reaction system.
                file_path: Absolute path to a pkl file.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ]
            
            if (not thread_id in this.reactions_systems) or \
                (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            with open(file_path, mode="wb") as f:
                obj = this.reactions_systems[thread_id][system_name]
                pickle.dump(obj, f)
            return f"Reaction system {system_name} is dumped to {file_path}."

        @tool 
        def load_from_file(system_name: str, file_path:str, runtime:ToolRuntime[Any]):
            '''
            Load a reaction system object from the pickle file and assign a name to it.
            Args:
                system_name: Name of the reaction system.
                file_path: Absolute path to the pickle file which contains a reaction system object.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ] 
            with open(file_path, mode="rb") as f:
                obj = pickle.load(f)
                if not thread_id in this.reactions_systems:
                    this.reactions_systems[thread_id] = dict()
                this.reactions_systems[thread_id][system_name] = obj 
            return f"ReactionSystem object is loaded from {file_path}, with key {system_name}." 

        @tool
        def has_reaction_system(system_name: str, runtime:ToolRuntime[Any])->bool:
            '''
            Cheack if a reaction system exist.
            Args:
                system_name: Name of the reaction system.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ]
            if not thread_id in this.reactions_systems:
                return False 
            if not system_name in this.reactions_systems[thread_id]:
                return False 
            return True

        @tool
        def delete_reaction_system(system_name: str, runtime:ToolRuntime[Any]) -> str:
            '''
            Delete a reaction system.
            Args:
                system_name: Name of the reaction system.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ] 
            if not thread_id in this.reactions_systems:
                return "No reaction system exists in the current thread."
            if not system_name in this.reactions_systems[thread_id]:
                return f"{system_name} does not exist in current thread."
            this.reactions_systems[thread_id].pop(system_name)
            return f"{system_name} is deleted from current thread."


        @tool 
        def create_reaction_system(
            runtime:ToolRuntime[Any],
            system_name: str,
            reactants_mol_pathes:List[str],
            products_mol_pathes:List[str],
            json_log_path: str,
            visualization_dir: str):
            '''
            Create a `ReactionSystem` object. A molecule should only appear one time. For example, for reaction "N2 + 3*H2 <-> 2*NH3", just call this tool by `reactants_mol_pathes=['/path/to/N2.xyz', '/path/to/H2.xyz']` and `products_mol_pathes = ['/path/to/NH3.xyz']`. 
            Args:
                system_name: A short name of the reaction system, e.g. 'CO2,CH4->CO,H2,H2O'.
                reactants_mol_pathes: Path to reactant molecules files (usually in `.xyz` format).
                products_mol_pathes: Path to product molecules files.
                json_log_path: Path to the json file which logs the state of `ReactionSystem` object.
                visualization_dir: The dir path to save visualization pictures of intermediate species and reactions.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ] 
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
        def render(runtime:ToolRuntime[Any], system_name: str, backend:Literal["ase", "rdkit"] = "rdkit"):
            '''
            Render the intermediates and reactions to pictures.
            Args:
                system_name: Name of the reaction system.
                backend: The backend used to render the reaction system.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ] 
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
            runtime:ToolRuntime[Any], 
            system_name: str,
            ntimes:int = -1,
            max_natoms_tot:int = -1, 
            min_natoms_tot:int = 0, 
            max_natoms:Dict[str, int] = None, 
            min_natoms:Dict[str, int] = None, 
            forbidden_pattern: List[Tuple[str, str]] = None):
            '''
            NOTE: You should firstly call `pause_to_ask` to ask the user to specify the args of this tool.

            Generate intermediate species and elementary steps satisfying certain constraints. Generate two files: a json file containing the information of reaction system and a `intermediates.extxyz` file containing all the species. The json file has 3 fields:
            - n_reactions: The number of reactions.
            - n_species: The number of intermediate species.
            - reactions: A dict list, contains information of each reaction.
                - id: Id of reaction.
                - reac_str: The string representation of reaction.
                - reactant_indices: The indices of reactants in `intermediates.extxyz`
                - product_indices: The indices of products.

            Args:
                system_name: Name of the reaction system, the corresponding `ReactionSystem` object should be created in advance.
                ntimes: `ntimes = -1` (default) use BFS to find all possible elementary steps. `ntimes > 0` sample the elementary steps n times using DFS (usually fail).
                max_natoms_tot: Max number of atoms in the intermediates. `-1` means no limitation.
                min_natoms_tot: Min number of atoms in the intermediates.
                max_natoms: A `dict` used to limit the max number of atoms of each element in the intermediates. If omitted, a default value is calculated from the final products.
                min_natoms: A `dict` used to limit the min number of atoms of each element in the intermediates.
                forbidden_pattern: A list of forbidden substructures represented by molecular graph, e.g., `('H,O,C,O,H','0-1,1-2,2-3,3-4')` represents geminal diol; `('C,O,O', '0-1,1-2')` represents a "C-O-O" sub-structure. If omitted, a list of default patterns are generated.
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ] 
            if (not thread_id in this.reactions_systems) or \
                  (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]
            flag, msg = rs.generate_elementary_steps(ntimes=ntimes,
                                         max_natoms_tot=max_natoms_tot,
                                         min_natoms_tot=min_natoms_tot,
                                         max_natoms=max_natoms,
                                         min_natoms=min_natoms,
                                         forbidden_pattern=forbidden_pattern)
            if not flag:
                return msg
            msg_dump = rs.dump_info(rs.json_log_path)
            return f"{msg}\n{msg_dump}\nCall `render` to visualize the reactions."


        @tool
        def delete_species(runtime:ToolRuntime[Any],
                           system_name: str, 
                           species_indices: List[int] = []):
            '''
            NOTE: You should firstly call `pause_to_ask`  to ask user for the species indices they want to delete.

            Delete intermediate speices from the system. Related or isolated reactions will also be deleted. Reactions and species indices will change after deletion. If you want to call this tool again, please tell the user new indices. Json log file and intermediates.extxyz will be updated.
            Args:
                system_name: Name of the reaction system.
                species_indices: Indices of intermediate speices.
            '''
            if not species_indices:
                return "No species indices indices are set, no intermediate speices are deleted."
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ]
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]
            rs.delete_species(species_indices)
            msg = rs.dump_info(rs.json_log_path)
            return f"Intermediates {species_indices} are deleted. {msg}"

        @tool
        def delete_reactions(runtime:ToolRuntime[Any],
                             system_name: str, 
                             reaction_indices: List[int] = []):
            '''
            NOTE: You should firstly call `pause_to_ask`  to ask user for the reaction indices they want to delete.

            Delete reactions from the system. If reactions which are not in `reaction_indices` become isolated, they will be automatically deleted, related intermediate species will also be deleted. Reactions and species indices will change after deletion. If you want to call this tool again, please tell the user new indices. Json log file and intermediates.extxyz will be updated.
            Args:
                system_name: Name of the reaction system.
                reaction_indices: Indices of reactions. 
            '''
            if not reaction_indices:
                return "No reaction indices are set, no reactions are deleted."
            
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ] 
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
        def merge_reactions(runtime:ToolRuntime[Any],
                            system_name: str, 
                            reaction_index_pairs: List[Tuple[int, int]] = []):
            '''
            NOTE: You should firstly call `pause_to_ask`  to let the user set reaction indices pairs they want to merge.

            Merge reactions in the system. Json log file and intermediates.extxyz will be updated.
            Args:
                system_name: Name of the reaction system.
                reaction_index_pairs: A list of tuples, each tuple contains the indices of reactions need to be merged. 
            ''' 
            if not reaction_index_pairs:
                return "No valid reaction pairs are set. No reactions are merged."
            
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ]
            if (not thread_id in this.reactions_systems) or \
                    (not system_name in this.reactions_systems[thread_id]):
                raise KeyError(f"{system_name} dose not exist. Check the system name or create it.")
            rs = this.reactions_systems[thread_id][system_name]

            for a, b in reaction_index_pairs:
                rs.merge_reactions(a, b)
            
            msg = rs.dump_info(rs.json_log_path)
            return f"Reaction pairs {reaction_index_pairs} are merged. Call `delete_reactions` to delete redundant reactions. {msg}"

        @tool 
        def bind_to_surface(runtime:ToolRuntime[Any],
                            system_name: str,
                            surface_structures_path:List[str],
                            cwd:str = None):
            '''
            Build initial and final states (in POSCAR format) of elementary reactions on the given surfaces. Atomic indices are automatically matched, fragments are aligned. 
            
            Structure files will be put in dirs `surface_<surface_index>/reaction_<reaction_index>` and will be named by `POSCAR_<state>_<conf_index>`. Here `<state>` is `ini` for `fin`, `<conf_index>` is the index of configuration.
            This tool also generate two json files:

            1. binding_info.json: this json has 3 fields:
                - surface_structure: Path to surface structure.
                - n_reactions: The number of reactions
                - reactions_info: A list of dict, each dict has 2 fields:
                    - reaction_string: String representation of a reaction.
                    - ads_atom_indices: The atomic indices of adsorbate atoms.
            2. intermediate_index.json: this json contains a dict, key is the smiles string of an intermediate species, value is a 3-elements tuple: (ir, 0/1, list) 
                - First element: The index of reaction which contains this species.
                - Second element: 0 or 1. 0 means this species is reactant, 1 means this species is product.
                - Third element: A list of int. These are the indices of structures, see below.
            Only species that appear alone on one side of at least one reaction are included. Species that only appear together with another fragment (i.e. co-adsorbed species) are not included, these are usually single atom adsorbates such as C, H and O, you can build them by `build_adsorption_configs_on_surface` tool. 
            Because there are usually more reactions than intermediates. For example, CO can appear in reactions `CO2<->CO+O`, `CO+H<->COH` and `CO<->C+O`. For these reactions, they can use the same CO adsorption structure. We provide intermediate_index.json for convenience, to only perform calculation on the adsorption structure once, instead of calculating it for each reaction. For example, for intermediate CO, suppose the tuple is (1, 0, [0,1,2]). Then we can only perform DFT calculation on IS (the second element is 0) structures 0,1,2 of reaction 1. These are all the possible CO adsorption structures we will meet in this reaction system.

            Args:
                system_name: Name of the reaction system.
                surface_structures_path: Pathes to surface structures, should be readable to ase.
                cwd: Current working directory. 
            '''
            stepnode_thread_id:str = runtime.config.get("configurable", {}).get("thread_id") 
            thread_id = stepnode_thread_id[0:stepnode_thread_id.find(':') ]
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
                    json.dump(info, f, indent=4)
                with open(f"surface_{i}/intermediate_index.json", mode="w") as f:
                    json.dump(intermediate_index_dict, f, indent=4) 
            if cwd:
                os.chdir(old_wd)

            return f"The POSCARs initial and final states are written to `surface_*` dirs in {cwd if cwd else old_wd}."

            
        return [dump, load_from_file, has_reaction_system, delete_reaction_system, 
                create_reaction_system, render, generate_elementary_steps, 
                delete_species, merge_reactions, delete_reactions, bind_to_surface]
    
