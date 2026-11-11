from ase import Atoms 
from typing import  List, Tuple
from mp_api.client import MPRester

def get_bulk_structure_from_mp(
        api_key:str,
        material_ids: List[str] = None,
        elements:str = None,
        formula:str = None) -> Tuple[List[Atoms], List[dict]]:
    """
    Get structures from materials project by material ids `material_ids`, elemental composition `elements` or chemical formula `formula`.    

    Args:
        api_key: Api key of materials project.
        material_ids: A list of material ids.
        elements: Elements separated by hyphen, e.g., "Ni-O", "Si-O".
        formula: Chemical formula of the system. 
        
    Returns:
        A list of `ase.Atoms` and a list of corresponding detailed informations containing the following fields:
    - `material_id`: Material id of this structures.
    - `formula`: Chemical formula.
    - `symmetry`: Symmetry information.
    - `nsites`: Number of sites.
    - `band_gap`: Band gap.
    - `energy_above_hull`: Energy above hull.
    - `formation_energy_per_atom`: Formation energy per atom
    - `total_magnetization`: Total Magnetization
    - `magnetic_ordering`: Magnetic ordering
        
    """

    fields = ["nsites", 
            "material_id", 
            "structure", 
            "formation_energy_per_atom", 
            "energy_above_hull", 
            "ordering", 
            "total_magnetization", 
            "band_gap", 
            "symmetry", 
            "formula_pretty"]
    

    with MPRester(api_key) as mpr:
        if material_ids:
            docs = mpr.materials.summary.search(material_ids = material_ids, fields=fields)
        elif elements:
            docs = mpr.materials.summary.search(chemsys = elements, fields=fields)
        elif formula:
            docs = mpr.materials.summary.search(formula = formula, fields=fields)
        else:
            raise ValueError("Error: `material_ids`, `elements` or `formula` must be set.")


    informations = []
    atoms_list = []

    for doc in docs:
        atoms_list.append(doc.structure.to_ase_atoms())
        informations.append(
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "symmetry": doc.symmetry.model_dump(),
                "nsites": doc.nsites,    
                "band_gap": doc.band_gap, 
                "energy_above_hull": doc.energy_above_hull, 
                "formation_energy_per_atom": doc.formation_energy_per_atom, 
                "total_magnetization": doc.total_magnetization,
                "magnetic_ordering": doc.ordering 
            }
        )
    return atoms_list, informations

        
    