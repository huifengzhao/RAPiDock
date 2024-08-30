##########################################################################
# File Name: dataset_utils.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Tue 24 Oct 2023 01:47:39 PM CST
#########################################################################


from rdkit import Chem
import MDAnalysis
from utils.PeptideBuilder import get_edges_from_sequence
import re
from Bio.PDB import PDBParser


def standard_residue_sort(item):
    # convert to str
    if isinstance(item, int):
        return item, 0
    else:
        s = str(item)
        # extract the digital part
        num = "".join([i for i in s if i.isdigit()])

        # extract the non digital part
        non_num = "".join([i for i in s if not i.isdigit()])
        code = ord(non_num)
        if num == "1":
            return (int(num) if num else 0, -code)
        else:
            return (int(num) if num else 0, code)


three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

three_to_one_esm = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",  #
    "SER": "S",
    "SEC": "U",  #
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",  #
    "GLX": "Z",  #
    "XAA": "X",  #
    "XLE": "J",
}  #


def read_pdb_with_connect_labels(
    pdbfile: str, sanitize: bool = True, addHs: bool = False
):
    mol = Chem.MolFromPDBFile(pdbfile, sanitize=False)

    rw_mol = Chem.RWMol(mol)
    while rw_mol.GetNumBonds() > 0:
        bond = rw_mol.GetBondWithIdx(0)
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    edges = set()
    for line in open(pdbfile, "r").readlines():
        if line.startswith("CONECT"):
            content = line.strip().split()[1:]
            start = content[0]
            ends = content[1:]
            for end in set(ends):
                edges.add((tuple(sorted((int(start), int(end)))) + (ends.count(end),)))

    for edge in edges:
        atom1_idx, atom2_idx, bond_type = edge
        if bond_type == 1:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.SINGLE)
        elif bond_type == 2:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.DOUBLE)
        elif bond_type == 3:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.TRIPLE)
        else:
            raise RuntimeError

    if sanitize:
        Chem.SanitizeMol(rw_mol)

    if addHs:
        mol = Chem.AddHs(rw_mol, addCoords=True)
    else:
        mol = rw_mol

    return mol


def read_pdb_with_seq(pdbfile: str, sanitize: bool = True, addHs: bool = False):
    mol = Chem.MolFromPDBFile(pdbfile, sanitize=False)

    rw_mol = Chem.RWMol(mol)
    while rw_mol.GetNumBonds() > 0:
        bond = rw_mol.GetBondWithIdx(0)
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    u = MDAnalysis.Universe(pdbfile)
    trans = {}
    seq = []
    for res_idx, residue in enumerate(u.residues):
        res_name = residue.resname.strip()
        seq.append(
            three_to_one[res_name]
            if res_name in three_to_one.keys()
            else f"[{res_name}]"
        )
        trans[
            (
                int(residue.resid)
                if residue.icode == ""
                else f"{residue.resid}{residue.icode.strip()}"
            )
        ] = res_idx
    real_idx = [
        trans[idx] for idx in sorted(set(trans.keys()), key=standard_residue_sort)
    ]
    seq = [seq[i] for i in real_idx]
    seq = "".join(seq)
    oxt = len(u.atoms.select_atoms("name OXT")) == 1
    edges = get_edges_from_sequence(seq, oxt).tolist()

    for edge in edges:
        atom1_idx, atom2_idx, bond_type = edge
        if bond_type == 1:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.SINGLE)
        elif bond_type == 2:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.DOUBLE)
        elif bond_type == 3:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.TRIPLE)
        else:
            raise RuntimeError

    if sanitize:
        Chem.SanitizeMol(rw_mol)

    if addHs:
        mol = Chem.AddHs(rw_mol, addCoords=True)
    else:
        mol = rw_mol

    return mol


def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq_pro = ""
        seq_dic = {}
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n, c, o = None, None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
                if atom.name == "O":
                    o = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None and o != None
            ):  # only append residue if it is an amino acid
                try:
                    seq_dic[
                        (
                            int(residue.id[1])
                            if residue.id[2] == " "
                            else f"{residue.id[1]}{residue.id[2].strip()}"
                        )
                    ] = three_to_one_esm[residue.get_resname()]
                except Exception as e:
                    seq_dic[
                        (
                            int(residue.id[1])
                            if residue.id[2] == " "
                            else f"{residue.id[1]}{residue.id[2].strip()}"
                        )
                    ] = "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        try:
            digit_list = [i for i in seq_dic.keys() if isinstance(i, int)]
            for idx in sorted(
                (
                    set(seq_dic.keys())
                    | set(range(min(digit_list), max(digit_list) + 1))
                    if len(digit_list) > 0
                    else set(seq_dic.keys())
                ),
                key=standard_residue_sort,
            ):
                try:
                    seq_pro += seq_dic[idx]
                except:
                    seq_pro += "-"
                    print(
                        "missed AA: ",
                        idx,
                        " in the complex ",
                        file_path,
                        ". Add it with a dash - .",
                    )
        except:
            print("=========================================" + file_path)

        if sequence is None:
            sequence = seq_pro
        else:
            sequence += ":" + seq_pro

    return sequence


def get_sequences(descriptions):
    new_sequences = []
    for description in descriptions:
        if "pdb" in description:
            new_sequences.append(get_sequences_from_pdbfile(description))
        else:
            new_sequences.append(re.sub(r"\[.*?\]", "-", description))
    return new_sequences
