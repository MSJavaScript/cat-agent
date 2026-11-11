from ase import Atoms
from ase.mep import NEB 

from typing import List 
from nebkit.tools.utils import get_structures_distance

def build_neb_path(ini_state:Atoms, 
                   fin_state:Atoms, 
                   image_spacing:float = 0.5,
                   n_images:int = None) -> List[Atoms]:
    '''
    Directly use idpp method to make NEB interpolation, without additional processing.
    Args:
        ini_state: `ase.Atoms` object of initial state.
        fin_state: `ase.Atoms` object of final state.
        image_spacing: Spacing between adjacent images. This argument is ignored if `n_images` is not `None`.
        n_images: The number of interpolation images.
    '''
    if n_images is None:
        # 计算初末态之间的距离
        d = get_structures_distance(ini_state, fin_state)
        n_images = int(d / image_spacing) - 1
    
    images = [ini_state]
    images += [ini_state.copy() for i in range(n_images)]
    images.append(fin_state)
    neb = NEB(images, method='improvedtangent')
    neb.interpolate(method='idpp')
    return neb.images