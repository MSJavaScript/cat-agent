from typing import List, Literal
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from ase.utils import rotate
from ase.visualize.plot import plot_atoms
from ase import Atoms 
import numpy as np

from PIL import Image
import io

from .utils import ase_atoms_to_mol

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw

def make_flat_2d(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol) 
    mol.RemoveAllConformers() 
    return mol

def _projected_bbox(atoms, rotation: str):
    """Bounding box (xlo, ylo, xhi, yhi) of the rotated molecule, in Angstrom.

    Uses the same radii (covalent) and the same projection (first two columns
    of ``positions @ rotate(rotation)``) as ``ase.visualize.plot.plot_atoms``,
    so the box exactly encloses what will be drawn.
    """
    # white space around each molecule, in Angstrom
    _PAD = 0.18
    R = rotate(rotation)
    P = atoms.positions @ R
    r = covalent_radii[atoms.numbers]
    return (P[:, 0].min() - r.max() - _PAD, P[:, 1].min() - r.max() - _PAD,
            P[:, 0].max() + r.max() + _PAD, P[:, 1].max() + r.max() + _PAD)


def visualize_reaction(reactants: List[Atoms], reactant_coeffs: List[int],
                       products: List[Atoms], product_coeffs: List[int],
                       width: int, height: int, save_name: str,
                       dpi: int = 75, rotation: str = "0x,90y,0z") -> plt.Figure:

    # fraction of the figure kept empty at the left/right edges
    _MARGIN_X = 0.02
    # fraction of the figure kept empty at the top/bottom
    _MARGIN_Y = 0.04
   
    if len(reactants) != len(reactant_coeffs) or len(products) != len(product_coeffs):
        raise ValueError("species and coefficient lists must have equal length")

    fig = plt.figure(figsize=(width, height))
    fontsize = min(9.0 * height, 6.0 * width, 30.0)
    em = fontsize / 72.0  # ~width of one character, in inches

    # Horizontal sequence of items: molecule images, '+' and '<->' symbols.
    items = []  # dicts: kind 'img' | 'sym'
    for i, (atoms, coeff) in enumerate(zip(list(reactants) + list(products),
                                          list(reactant_coeffs) + list(product_coeffs))):
        if coeff < 1:
            raise ValueError("coefficients must be >= 1")
        items.append({"kind": "img", "coeff": coeff, "atoms":atoms})
        if i == len(reactants) - 1:
            items.append({"kind": "sym", "text": "\u2194", "w": 1.7 * em})
        elif i < len(reactants) + len(products) - 1:
            items.append({"kind": "sym", "text": "+", "w": 1.3 * em})

    for item in items:
        if item["kind"] == "img":
            xlo, ylo, xhi, yhi = _projected_bbox(item["atoms"], rotation)
            item["bbox"] = (xlo, ylo, xhi, yhi)
            item["ext_x"] = xhi - xlo
            item["ext_y"] = yhi - ylo
            item["label_w"] = (0.62 * len(f"{item['coeff']}*") + 0.1) * em \
                if item["coeff"] > 1 else 0.0

    # One shared scale (inches per Angstrom) for every molecule, so all
    # species are drawn at the same physical size.
    avail_w = width * (1.0 - 2.0 * _MARGIN_X)
    avail_h = height * (1.0 - 2.0 * _MARGIN_Y)
    fixed_w = sum(it.get("w", 0.0) + it.get("label_w", 0.0) for it in items)
    scale = min((avail_w - fixed_w) / sum(it["ext_x"] for it in items
                                          if it["kind"] == "img"),
                avail_h / max(it["ext_y"] for it in items if it["kind"] == "img"))

    # Place items left to right, all vertically centered.
    x = (width - (fixed_w + scale * sum(it["ext_x"] for it in items
                                        if it["kind"] == "img"))) / 2.0
    for item in items:
        if item["kind"] == "sym":
            fig.text((x + item["w"] / 2.0) / width, 0.5, item["text"],
                     ha="center", va="center", fontsize=fontsize)
            x += item["w"]
            continue

        w_in = scale * item["ext_x"]
        if item["coeff"] > 1:
            fig.text((x + item["label_w"] - 0.12 * em) / width, 0.5,
                     f"{item['coeff']}*",
                     ha="right", va="center", fontsize=fontsize)
        x += item["label_w"]

        h_in = scale * item["ext_y"]
        ax = fig.add_axes([x / width,
                           (height - h_in) / 2.0 / height,
                           w_in / width, h_in / height])
        # scale=1 makes the axes data coordinates Angstrom, and bbox places
        # the molecule exactly inside its box, so every molecule is drawn
        # at the shared inches-per-Angstrom scale set by the axes size.
        xlo, ylo, xhi, yhi = item["bbox"]
        plot_atoms(item["atoms"], ax, rotation=rotation, scale=1.0,
                   bbox=(xlo, ylo, xhi, yhi), show_unit_cell=0)
        x += w_in

    fig.savefig(save_name, dpi=dpi)
    return fig


def _species_to_mol(atoms: Atoms, skin: float, ignoreH: bool) -> Chem.Mol:
    """spec -> RDKit mol (bare graph, no conformer) for reaction drawing."""
    mol = ase_atoms_to_mol(atoms, skin=skin, ignoreH=ignoreH)
    with rdBase.BlockLogs():
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # over-coordinated single-bond structures fail valence checks;
            # the drawer copes without full sanitization
            mol.UpdatePropertyCache(strict=False)
        # DrawReaction computes clean 2D depictions itself for conformer-free
        # mols; mols carrying a conformer (e.g. the ETKDG one from
        # ase_atoms_to_mol) trip a layout bug that collapses species like H2
        # to an empty image, so hand over the bare graph.
        mol.RemoveAllConformers()
    return mol

def visualize_reaction_rdkit(reactants: List[Chem.Mol], reactant_coeffs: List[int],
                             products: List[Chem.Mol], product_coeffs: List[int],
                             width: int, height: int, save_name: str,
                             dpi: int = 75) -> rdChemReactions.ChemicalReaction:
   
    if len(reactants) != len(reactant_coeffs) or len(products) != len(product_coeffs):
        raise ValueError("species and coefficient lists must have equal length")
    if any(c < 1 for c in list(reactant_coeffs) + list(product_coeffs)):
        raise ValueError("coefficients must be >= 1")
    with rdBase.BlockLogs():
        rxn = rdChemReactions.ChemicalReaction()
        for mol, coeff in zip(reactants, reactant_coeffs):
            #mol = _species_to_mol(atoms, skin, ignoreH)
            for _ in range(coeff):
                rxn.AddReactantTemplate(make_flat_2d(mol))
        for mol, coeff in zip(products, product_coeffs):
            #mol = _species_to_mol(atoms, skin, ignoreH)
            for _ in range(coeff):
                rxn.AddProductTemplate(make_flat_2d(mol))
        rxn.Initialize()

        drawer = rdMolDraw2D.MolDraw2DCairo(int(width * dpi), int(height * dpi))
        drawer.drawOptions().includeRadicals = False
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
    with open(save_name, "wb") as fh:
        fh.write(drawer.GetDrawingText())
    return rxn


def draw_molecules_array(atoms_list: List[Atoms], 
                         mols_list: List[Chem.Mol],
                         nrow: int, ncol: int,
                         width: float, height: float, save_name:str, dpi: int = 75,
                         backend: Literal["ase", "rdkit"] = "ase",
                         rotation: str = "0x,90y,0z") -> plt.Figure:
    _CELL_PAD = 0.05
    if backend not in ("ase", "rdkit"):
        raise ValueError(f"unknown backend {backend!r}; use 'ase' or 'rdkit'")
    if len(atoms_list) > nrow * ncol:
        raise ValueError(f"{len(atoms_list)} molecules do not fit "
                         f"in a {nrow}x{ncol} grid")

    fig = plt.figure(figsize=(width, height))
    cell_w, cell_h = width / ncol, height / nrow
    # index label in the bottom strip of each cell, sized with the cell
    label_fs = min(0.22 * cell_h, 0.12 * cell_w, 20.0 / 72.0) * 72.0  # pt
    label_h = 1.3 * label_fs / 72.0  # inches
    img_w = cell_w * (1.0 - 2.0 * _CELL_PAD)
    img_h = cell_h * (1.0 - 2.0 * _CELL_PAD) - label_h

    if backend == "ase":
        bboxes = [_projected_bbox(atoms, rotation) for atoms in atoms_list]
        # one shared scale, so e.g. an O atom cell is smaller than a CO2 cell
        scale = min(img_w / max(b[2] - b[0] for b in bboxes),
                    img_h / max(b[3] - b[1] for b in bboxes))
    else:
        mols = [make_flat_2d(item_mol) for item_mol in mols_list]
        #mols = [_species_to_mol(atoms, skin, ignoreH) for atoms in atoms_list]

    for i, atoms in enumerate(atoms_list):
        r, c = divmod(i, ncol)
        x0 = c * cell_w                  # cell left edge, inches
        y0 = height - (r + 1) * cell_h   # cell bottom edge, inches
        fig.text((x0 + cell_w / 2.0) / width, (y0 + label_h / 2.0) / height,
                 str(i), ha="center", va="center", fontsize=label_fs)

        if backend == "ase":
            w_in = scale * (bboxes[i][2] - bboxes[i][0])
            h_in = scale * (bboxes[i][3] - bboxes[i][1])
            ax = fig.add_axes([(x0 + (cell_w - w_in) / 2.0) / width,
                               (y0 + label_h + (img_h - h_in) / 2.0) / height,
                               w_in / width, h_in / height])
            # scale=1 + bbox draws the molecule exactly inside its box
            plot_atoms(atoms, ax, rotation=rotation, scale=1.0,
                       bbox=bboxes[i], show_unit_cell=0)
        else:
            ax = fig.add_axes([(x0 + cell_w * _CELL_PAD) / width,
                               (y0 + label_h) / height,
                               img_w / width, img_h / height])
            drawer = rdMolDraw2D.MolDraw2DCairo(int(img_w * dpi),
                                                int(img_h * dpi))
            drawer.drawOptions().includeRadicals = False
            drawer.DrawMolecule(mols[i])
            drawer.FinishDrawing()
            img = Image.open(io.BytesIO(drawer.GetDrawingText()))
            ax.imshow(np.asarray(img))
            ax.axis("off")

            # ax = fig.add_axes([(x0 + cell_w * _CELL_PAD) / width,
            #                    (y0 + label_h) / height,
            #                    img_w / width, img_h / height])
            # img = Draw.MolToImage(mols[i], size=(int(img_w * dpi), int(img_h * dpi)), includeRadicals=False)
            # ax.imshow(np.asarray(img))
            # ax.axis("off")

    fig.savefig(save_name, dpi=dpi)
    return fig