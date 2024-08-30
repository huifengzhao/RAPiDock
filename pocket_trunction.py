##########################################################################
# File Name: pocket_trunction.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:31:15 AM CST
#########################################################################

"""
trunction of protein under three different level: Chain, Residue, Atom
"""

from Bio.PDB import NeighborSearch, PDBIO, Select, PDBParser
import argparse
import numpy as np


class PocketResidueSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues


def residues_saver(structure, residues, out_name, verbose=0):
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(out_name, PocketResidueSelect(residues))
    if verbose:
        print("residues saved at {}".format(out_name))

def pocket_trunction(
    protein,
    peptide=None,
    threshold=20.0,
    save_name=None,
    xyz=None,
    level="Chain",
    exclude_chain=None,
    threshold_keep=5.0,
):
    # Load Peptide Structures
    if peptide is not None:
        parser = PDBParser()
        peptide_structure = parser.get_structure("peptide", peptide)
        peptide_coords = [atom.get_coord() for atom in peptide_structure.get_atoms()]
    if xyz is not None:
        peptide_coords = [np.array(xyz)]

    other_chain = set(exclude_chain) if exclude_chain is not None else {}

    # Load Protein Sructures
    if type(protein) == str:
        parser = PDBParser()
        protein_structure = parser.get_structure("protein", protein)

    # Extract pocket residues
    ns = NeighborSearch(list(protein_structure.get_atoms()))
    pocket_residues_far = {
        res.get_parent()
        for coord in peptide_coords
        for res in ns.search(coord, threshold)
    }  # 5-20A 
    pocket_residues_near = {
        res.get_parent()
        for coord in peptide_coords
        for res in ns.search(coord, threshold_keep)
    }  # 0-5-A 
    pocket_chains_far = {residue.get_parent() for residue in pocket_residues_far}
    pocket_chains_near = {residue.get_parent() for residue in pocket_residues_near}

    pocket_residues = []
    if level == "Residue":
        pocket_residue_list_far = [
            [
                residue
                for residue in pocket_residues_far
                if residue.get_parent() == chain
            ]
            for chain in pocket_chains_far
        ]
        pocket_residue_list_near = [
            [
                residue
                for residue in pocket_residues_near
                if residue.get_parent() == chain
            ]
            for chain in pocket_chains_near
        ]

        for _ in pocket_residue_list_near:
            chain_id = _[0].get_parent()
            res_ids = [res.get_full_id()[-1][1] for res in _]
            res_id_min = min(res_ids)
            res_id_max = max(res_ids)
            # print(chain_id, res_id_min,res_id_max)
            for chain in protein_structure.get_chains():
                if chain == chain_id:
                    for residue in chain.get_residues():
                        if res_id_min <= residue.get_full_id()[-1][1] <= res_id_max:
                            pocket_residues.append(residue)
        for _ in pocket_residue_list_far:
            if (_[0].get_parent().id in other_chain) and (
                _[0].get_parent() in list(pocket_chains_near)
            ):
                continue
            chain_id = _[0].get_parent()
            res_ids = [res.get_full_id()[-1][1] for res in _]
            res_id_min = min(res_ids)
            res_id_max = max(res_ids)
            # print(chain_id, res_id_min,res_id_max)
            for chain in protein_structure.get_chains():
                if chain == chain_id:
                    for residue in chain.get_residues():
                        if res_id_min <= residue.get_full_id()[-1][1] <= res_id_max:
                            pocket_residues.append(residue)

    elif level == "Chain":
        for chain in protein_structure.get_chains():
            if chain in list(pocket_chains_near):
                for residue in chain.get_residues():
                    pocket_residues.append(residue)
            if (
                (chain in list(pocket_chains_far))
                and (chain.id not in other_chain)
                and (chain not in list(pocket_chains_near))
            ):
                for residue in chain.get_residues():
                    pocket_residues.append(residue)

    elif level == "Atom":
        pocket_residues = list(pocket_residues_far) + list(pocket_residues_near)

    # (Optional): save the pocket resides
    if save_name:
        residues_saver(protein_structure, pocket_residues, save_name)
    return len([res for res in protein_structure.get_residues()]), len(pocket_residues)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--protein_path", type=str, default=None, help="Path to the protein file"
)
parser.add_argument(
    "--peptide_path", type=str, default=None, help="Path to the peptide file"
)
parser.add_argument(
    "--xyz", nargs="+", type=float, default=None, help="Center of pocket"
)
parser.add_argument("--threshold", type=float, default=None, help="Cutoff threshold")
parser.add_argument(
    "--save_name", type=str, default=None, help="Path to the output protein file"
)
parser.add_argument("--level", type=str, default=None, help="Chain or Residue")
parser.add_argument(
    "--exclude_chain", nargs="+", type=str, default=None, help="Chain excluded"
)
parser.add_argument(
    "--threshold_keep",
    type=float,
    default=5,
    help="Cutoff threshold for keeping chains even they are included in exclude_chain",
)
args = parser.parse_args()

pocket_trunction(
    args.protein_path,
    args.peptide_path,
    args.threshold,
    args.save_name,
    args.xyz,
    args.level,
    args.exclude_chain,
    args.threshold_keep,
)
