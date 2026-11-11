"""Convert extended XYZ (extxyz) to DeePMD-kit HDF5 format.

Usage:
    python3 convert.py structs.extxyz data.hdf5
"""

from __future__ import annotations
import h5py
import copy
import hashlib
import sys
import ase.io
from ase.calculators.calculator import PropertyNotImplementedError
import numpy as np

# Frame properties dumped to HDF5: (dpdata key, deepmd file name, skip if nopbc)
FRAME_PROPERTIES = [
    ("cells", "box", True),
    ("coords", "coord", False),
    ("real_atom_types", "real_atom_types", False),
    ("energies", "energy", False),
    ("forces", "force", False),
    ("virials", "virial", False),
    ("atom_pref", "atom_pref", False),
]


def read_frames(file_name: str) -> list:
    """Read all frames of an extxyz file as ase.Atoms (same as dpdata's
    ``from_multi_systems`` of the ase/structure format)."""
    

    return ase.io.read(file_name, index=slice(None))


def atoms_to_data(atoms) -> dict:
    """Convert one ase.Atoms to a dpdata-style labeled data dict.

    Extracted from dpdata/plugins/ase.py ASEStructureFormat.
    """
    

    symbols = atoms.get_chemical_symbols()
    atom_names = list(dict.fromkeys(symbols))
    atom_numbs = [symbols.count(symbol) for symbol in atom_names]
    atom_types = np.array([atom_names.index(symbol) for symbol in symbols]).astype(
        int
    )
    data = {
        "atom_names": atom_names,
        "atom_numbs": atom_numbs,
        "atom_types": atom_types,
        "cells": np.array([atoms.cell[:]]),
        "coords": np.array([atoms.get_positions()]),
        "orig": np.zeros(3),
        "nopbc": not np.any(atoms.get_pbc()),
    }
    try:
        energies = atoms.get_potential_energy(force_consistent=True)
    except PropertyNotImplementedError:
        energies = atoms.get_potential_energy()
    forces = atoms.get_forces()
    data["energies"] = np.array([energies])
    data["forces"] = np.array([forces])

    # try to get virials from the virial field, or converted from stress
    virials = atoms.info.get("virial")
    if virials is None:
        try:
            stress = atoms.get_stress(voigt=False)
        except PropertyNotImplementedError:
            pass
        else:
            virials = -atoms.get_volume() * stress
    if virials is not None:
        data["virials"] = np.array([virials])
    return data


def sort_atom_names(data: dict, type_map: list | None = None) -> dict:
    """Reorder atom_names/atom_numbs/atom_types (dpdata.utils.sort_atom_names).

    Without type_map, names are sorted alphabetically; with type_map, names
    are assigned to type_map (zero-count entries allowed).
    """
    if type_map is not None:
        orig_names = data["atom_names"]
        orig_numbs = data["atom_numbs"]
        active_names = {name for name, numb in zip(orig_names, orig_numbs) if numb > 0}
        missing = active_names - set(type_map)
        if missing:
            raise ValueError(f"Active atom types {missing} not in provided type_map.")
        name_to_new_idx = {name: i for i, name in enumerate(type_map)}
        new_names = list(type_map)
        new_numbs = [0] * len(type_map)
        lookup = np.full(len(orig_names), -1, dtype=np.int64)
        for old_idx, name in enumerate(orig_names):
            new_idx = name_to_new_idx.get(name)
            if new_idx is not None:
                lookup[old_idx] = new_idx
                new_numbs[new_idx] = orig_numbs[old_idx]
        data["atom_types"] = lookup[np.asarray(data["atom_types"])]
        data["atom_names"] = new_names
        data["atom_numbs"] = new_numbs
    else:
        idx = np.argsort(data["atom_names"], kind="stable")
        data["atom_names"] = list(np.array(data["atom_names"])[idx])
        data["atom_numbs"] = list(np.array(data["atom_numbs"])[idx])
        inv_idx = np.argsort(idx, kind="stable")
        data["atom_types"] = inv_idx[data["atom_types"]]
    return data


def add_atom_names(data: dict, atom_names: list) -> dict:
    """Add atom_names that do not exist (dpdata.utils.add_atom_names)."""
    data["atom_names"].extend(atom_names)
    data["atom_numbs"].extend([0 for _ in atom_names])
    return data


def sort_atom_types(data: dict) -> dict:
    """Stable-sort atoms by type (dpdata System.sort_atom_types)."""
    idx = np.argsort(data["atom_types"], kind="stable")
    data["atom_types"] = data["atom_types"][idx]
    for key, _, _ in FRAME_PROPERTIES:
        if key in data and data[key].ndim >= 2:
            data[key] = data[key][:, idx]
    return data


def get_formula(data: dict) -> str:
    """Formula like C1H1Rh16, in atom_names order (zero counts included)."""
    return "".join(
        f"{symbol}{numb}"
        for symbol, numb in zip(data["atom_names"], data["atom_numbs"])
    )


def get_short_formula(data: dict) -> str:
    """Formula with zero-count elements removed."""
    return "".join(
        f"{symbol}{numb}"
        for symbol, numb in zip(data["atom_names"], data["atom_numbs"])
        if numb
    )


def get_short_name(data: dict) -> str:
    """Short name of the system (dpdata System.short_name), limited to
    255 bytes: formula, then short_formula, then the formula hash."""
    formula = get_formula(data)
    if len(formula.encode("utf-8")) <= 255:
        return formula
    short_formula = get_short_formula(data)
    if len(short_formula.encode("utf-8")) <= 255:
        return short_formula
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()


def append_system(data: dict, other: dict) -> dict:
    """Concatenate frames (dpdata System.append for the same formula)."""
    if (other["atom_types"] != data["atom_types"]).any():
        # allow to append a system with a different atom_types order
        other = sort_atom_types(copy.deepcopy(other))
        data = sort_atom_types(data)
    for key, _, _ in FRAME_PROPERTIES:
        in_data, in_other = key in data, key in other
        if in_data != in_other:
            raise RuntimeError(
                f"systems with inconsistent {key} could not be appended"
            )
        if in_data:
            data[key] = np.concatenate((data[key], other[key]), axis=0)
    if data.get("nopbc") and not other.get("nopbc"):
        data["nopbc"] = False
    return data


def build_multi_systems(frames: list) -> dict[str, dict]:
    """Group frames by composition (dpdata.MultiSystems append logic).

    Returns a dict of formula -> data dict. All systems share the same
    atom_names (the global type map, in order of first appearance).
    """
    systems: dict[str, dict] = {}
    atom_names: list[str] = []
    for atoms in frames:
        data = atoms_to_data(atoms)
        sort_atom_names(data)
        # dpdata MultiSystems.check_atom_names
        new_in_system = [e for e in data["atom_names"] if e not in atom_names]
        if new_in_system:
            atom_names.extend(new_in_system)
            for each_system in systems.values():
                add_atom_names(each_system, new_in_system)
                sort_atom_names(each_system, type_map=atom_names)
        new_in_self = [e for e in atom_names if e not in data["atom_names"]]
        if new_in_self:
            add_atom_names(data, new_in_self)
        sort_atom_names(data, type_map=atom_names)

        formula = get_formula(data)
        if not formula:
            # ignore empty frames
            continue
        if formula in systems:
            systems[formula] = append_system(systems[formula], data)
        else:
            systems[formula] = copy.deepcopy(data)
    return systems


def dump_system(
    g,
    data: dict,
    set_size: int = 5000,
    comp_prec: np.dtype = np.float64,
) -> None:
    """Dump one system into a HDF5 group (dpdata formdats/deepmd/hdf5 dump)."""
    # ignore empty systems
    if not len(data["coords"]):
        return
    g.create_dataset("type.raw", data=data["atom_types"])
    g.create_dataset("type_map.raw", data=np.array(data["atom_names"], dtype="S"))
    nframes = data["cells"].shape[0]
    nopbc = data.get("nopbc", False)

    reshaped_data = {}
    for key, deepmd_name, skip_if_nopbc in FRAME_PROPERTIES:
        if key not in data:
            continue
        if skip_if_nopbc and nopbc:
            continue
        arr = np.asarray(data[key])
        if nframes > 0 and arr.size == 0:
            # an optional frame property may be empty; skip it
            continue
        ddata = np.reshape(arr, (nframes, -1))
        if np.issubdtype(ddata.dtype, np.floating):
            ddata = ddata.astype(comp_prec)
        reshaped_data[deepmd_name] = ddata

    nsets = nframes // set_size
    if set_size * nsets < nframes:
        nsets += 1
    for ii in range(nsets):
        set_stt = ii * set_size
        set_end = (ii + 1) * set_size
        set_folder = g.create_group("set.%03d" % ii)
        for fn, ddata in reshaped_data.items():
            set_folder.create_dataset(f"{fn}.npy", data=ddata[set_stt:set_end])

    if nopbc:
        g.create_dataset("nopbc", data=True)


def convert_extxyz_to_hdf5(
    file_name: str,
    out_file_name: str,
    set_size: int = 5000,
    comp_prec: np.dtype = np.float64,
) -> dict[str, dict]:
    """Convert an extxyz file to a DeePMD-kit HDF5 file.

    Parameters
    ----------
    file_name : str
        path to the input extxyz file
    out_file_name : str
        path to the output HDF5 file
    set_size : int, default 5000
        maximum number of frames per set.* subgroup
    comp_prec : np.dtype, default np.float64
        precision of floating point data

    Returns
    -------
    dict
        formula -> data dict of the converted systems
    """
    

    frames = read_frames(file_name)
    systems = build_multi_systems(frames)
    with h5py.File(out_file_name, "w") as f:
        for data in systems.values():
            g = f.create_group(get_short_name(data))
            dump_system(g, data, set_size=set_size, comp_prec=comp_prec)
    return systems


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <input.extxyz> <output.hdf5>")
        sys.exit(1)
    convert_extxyz_to_hdf5(sys.argv[1], sys.argv[2])
    print(f"Convert {sys.argv[1]} to {sys.argv[2]} successfully.")


if __name__ == "__main__":
    main()
