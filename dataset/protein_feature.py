##########################################################################
# File Name: protein_feature.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:20:49 AM CST
#########################################################################

## added by huifeng, 20231010

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import MDAnalysis
from scipy.spatial import distance_matrix
from MDAnalysis.analysis import distances
from utils.dataset_utils import standard_residue_sort

three2idx = {
    k: v
    for v, k in enumerate(
        [
            "GLY",
            "ALA",
            "VAL",
            "LEU",
            "ILE",
            "PRO",
            "PHE",
            "TYR",
            "TRP",
            "SER",
            "THR",
            "CYS",
            "MET",
            "ASN",
            "GLN",
            "ASP",
            "GLU",
            "LYS",
            "ARG",
            "HIS",
            "MSE",
            "X",
        ]
    )
}
three2self = {
    v: v
    for v in [
        "GLY",
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "PRO",
        "PHE",
        "TYR",
        "TRP",
        "SER",
        "THR",
        "CYS",
        "MET",
        "ASN",
        "GLN",
        "ASP",
        "GLU",
        "LYS",
        "ARG",
        "HIS",
        "MSE",
    ]
}

aa2tip = [
    "CA",  # gly
    "CB",  # ala
    "CB",  # val
    "CG",  # leu
    "CD1",  # ile
    "CG",  # pro
    "CZ",  # phe
    "OH",  # tyr
    "CH2",  # trp
    "OG",  # ser
    "OG1",  # thr
    "SG",  # cys
    "SD",  # met
    "ND2",  # asn
    "NE2",  # gln
    "CG",  # asp
    "CD",  # glu
    "NZ",  # lys
    "CZ",  # arg
    "NE2",  # his
    "SE",  # mse
    "CB",  # unknown (gap etc)
]
RES_MAX_NATOMS = 24


def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0, 0]
    for idx, angle in enumerate(
        [res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]
    ):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis


def calc_dist(res1, res2):
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def obatin_edge(res_lis, src, dst):
    dist = calc_dist(res_lis[src], res_lis[dst])
    return dist.min() * 0.1, dst.max() * 0.1


def check_connect(res_lis, i, j):
    if abs(i - j) == 1 and res_lis[i].segid == res_lis[j].segid:
        return 1
    else:
        return 0


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(n, ca, c):
    c, n = _normalize(c - ca), _normalize(n - ca)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def positional_embeddings_v1(edge_index, num_embeddings=16, period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    # raw
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    # new
    max_relative_feature = 32
    d = torch.clip(d + max_relative_feature, 0, 2 * max_relative_feature)
    d_onehot = F.one_hot(d, 2 * max_relative_feature + 1)
    E = torch.cat((torch.cos(angles), torch.sin(angles), d_onehot), -1)
    return E


def get_protein_feature_mda(receptor_file, top_k=24, lm_embedding_chains=None):
    protein_mol = MDAnalysis.Universe(receptor_file)
    with torch.no_grad():
        coords = []
        c_alpha_coords = []
        n_coords = []
        c_coords = []
        o_coords = []
        tip_coords = []
        valid_chain_ids = []
        valid_lm_embeddings = []
        lengths = []
        pure_res_lis, seq, node_s = [], [], []

        for i, chain in enumerate(protein_mol.segments):
            chain_coords = []  # num_residues, num_atoms, 3
            chain_c_alpha_coords = []
            chain_n_coords = []
            chain_c_coords = []
            chain_o_coords = []
            chain_tip_coords = []
            count = 0
            chain_seq = []
            chain_pure_res_lis = []
            chain_node_s = []
            trans = {}
            _ = -1
            lm_embedding_chain = (
                lm_embedding_chains[i] if lm_embedding_chains is not None else None
            )
            for res_idx, residue in enumerate(chain.residues):
                tip = None
                residue_coords = None
                c_alpha, n, c, o = None, None, None, None
                res_name = residue.resname.strip()
                res_atoms = residue.atoms.select_atoms("not type H")
                dists = distances.self_distance_array(res_atoms.positions)
                c_alpha = res_atoms.select_atoms("name CA")
                n = res_atoms.select_atoms("name N")
                c = res_atoms.select_atoms("name C")
                o = res_atoms.select_atoms("name O")
                residue_coords = res_atoms.positions
                if len(c_alpha) == 1 and len(n) == 1 and len(c) == 1 and len(o) == 1:
                    # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                    _ += 1
                    chain_c_alpha_coords.append(c_alpha.positions[0])
                    chain_n_coords.append(n.positions[0])
                    chain_c_coords.append(c.positions[0])
                    chain_o_coords.append(o.positions[0])
                    chain_coords.append(residue_coords)
                    intra_dis = [
                        dists.max() * 0.1,
                        dists.min() * 0.1,
                        distances.dist(c_alpha, o)[-1][0] * 0.1,
                        distances.dist(o, n)[-1][0] * 0.1,
                        distances.dist(n, c)[-1][0] * 0.1,
                    ]

                    tip = res_atoms.select_atoms(
                        f"name {aa2tip[three2idx[three2self.get(res_name, 'X')]]}"
                    )
                    if len(tip) == 0:
                        tip = res_atoms.select_atoms("name CA")
                    chain_tip_coords.append(tip.positions[0])

                    chain_seq.append(three2idx[three2self.get(res_name, "X")])
                    chain_node_s.append(intra_dis + obtain_dihediral_angles(residue))
                    chain_pure_res_lis.append(residue)

                    trans[
                        (
                            int(residue.resid)
                            if residue.icode == ""
                            else f"{residue.resid}{residue.icode.strip()}"
                        )
                    ] = _
                    count += 1
            # standardize the order of residues
            real_idx = [
                trans[idx]
                for idx in sorted(set(trans.keys()), key=standard_residue_sort)
            ]

            chain_coords = [chain_coords[i] for i in real_idx]
            chain_c_alpha_coords = [chain_c_alpha_coords[i] for i in real_idx]
            chain_n_coords = [chain_n_coords[i] for i in real_idx]
            chain_c_coords = [chain_c_coords[i] for i in real_idx]
            chain_o_coords = [chain_o_coords[i] for i in real_idx]
            chain_tip_coords = [chain_tip_coords[i] for i in real_idx]
            chain_seq = [chain_seq[i] for i in real_idx]
            chain_pure_res_lis = [chain_pure_res_lis[i] for i in real_idx]
            chain_node_s = [chain_node_s[i] for i in real_idx]

            embedding_idx = sorted(
                set(trans.keys())
                | set(
                    range(
                        min([i for i in trans.keys() if isinstance(i, int)]),
                        max([i for i in trans.keys() if isinstance(i, int)]) + 1,
                    )
                ),
                key=standard_residue_sort,
            )
            assert len(embedding_idx) == len(lm_embedding_chain)
            lm_embedding_chain = (
                [
                    lm_embedding_chain[i]
                    for i in [
                        idx for idx, i in enumerate(embedding_idx) if i in trans.keys()
                    ]
                ]
                if lm_embedding_chains is not None
                else None
            )

            lengths.append(count)
            coords.append(chain_coords)
            c_alpha_coords.append(np.array(chain_c_alpha_coords))
            n_coords.append(np.array(chain_n_coords))
            c_coords.append(np.array(chain_c_coords))
            o_coords.append(np.array(chain_o_coords))
            tip_coords.append(np.array(chain_tip_coords))
            seq.append(chain_seq)
            node_s.append(chain_node_s)
            pure_res_lis.append(chain_pure_res_lis)
            valid_lm_embeddings.append(lm_embedding_chain)
            if not count == 0:
                valid_chain_ids.append(chain.segid)

        coords = [
            atomlist
            for chainlist in coords
            for reslist in chainlist
            for atomlist in reslist
        ]  # list with n_residues arrays: [n_atoms, 3]
        c_alpha_coords = np.concatenate(c_alpha_coords, axis=0)  # [n_residues, 3]
        n_coords = np.concatenate(n_coords, axis=0)  # [n_residues, 3]
        c_coords = np.concatenate(c_coords, axis=0)  # [n_residues, 3]
        tip_coords = np.concatenate(tip_coords, axis=0)  # [n_residues, 3]
        lm_embeddings = (
            np.concatenate(valid_lm_embeddings, axis=0)
            if lm_embedding_chains is not None
            else None
        )
        seq = np.concatenate(seq, axis=0)
        node_s = np.concatenate(node_s, axis=0)  # [n_residues, 9]
        assert sum(lengths) == len(c_alpha_coords)
        # node features
        seq = torch.from_numpy(np.asarray(seq))  # 残基类型的整数编码
        node_s = torch.from_numpy(np.asarray(node_s))
        # edge features
        c_alpha_coords = torch.from_numpy(c_alpha_coords)
        n_coords = torch.from_numpy(n_coords)
        c_coords = torch.from_numpy(c_coords)
        tip_coords = torch.from_numpy(tip_coords)
        X_center_of_mass = torch.from_numpy(
            np.concatenate(
                [
                    residue.atoms.select_atoms("not type H").center_of_mass(
                        compound="residues"
                    )
                    for chain in pure_res_lis
                    for residue in chain
                ]
            )
        )  # [n_residues, 3]
        side_chain_mass = torch.from_numpy(
            np.asarray(
                [
                    (
                        residue.atoms.select_atoms(
                            "not name C N CA O and not type H"
                        ).center_of_mass()
                        if len(
                            residue.atoms.select_atoms(
                                "not name C N CA O and not type H"
                            )
                        )
                        > 0
                        else residue.atoms.select_atoms("name CA").positions[0]
                    )
                    for chain in pure_res_lis
                    for residue in chain
                ]
            )
        )  # [n_residues, 3]
        edge_index = torch_cluster.knn_graph(c_alpha_coords, k=top_k)
        dis_minmax = torch.from_numpy(
            np.asarray(
                [
                    obatin_edge(
                        [residue for chain in pure_res_lis for residue in chain],
                        src,
                        dst,
                    )
                    for src, dst in edge_index.T
                ]
            )
        ).view(
            edge_index.size(1), 2
        )  # [n_edges, 2]
        dis_matx_center = distance_matrix(X_center_of_mass, X_center_of_mass)
        dis_matx_side_center = distance_matrix(side_chain_mass, side_chain_mass)
        cadist = (
            torch.pairwise_distance(
                c_alpha_coords[edge_index[0]], c_alpha_coords[edge_index[1]]
            )
            * 0.1
        ).view(
            -1, 1
        )  # [n_edges, 1]
        cedist = (
            torch.from_numpy(dis_matx_center[edge_index[0, :], edge_index[1, :]]) * 0.1
        ).view(
            -1, 1
        )  # distance between two centers of mass # [n_edges, 1]
        csdist = (
            torch.from_numpy(dis_matx_side_center[edge_index[0, :], edge_index[1, :]])
            * 0.1
        ).view(
            -1, 1
        )  # distance between two centers of sidechain mass # [n_edges, 1]
        edge_connect = torch.from_numpy(
            np.asarray(
                [
                    check_connect(
                        [residue for chain in pure_res_lis for residue in chain], x, y
                    )
                    for x, y in edge_index.T
                ]
            )
        ).view(
            -1, 1
        )  # [n_edges, 1]
        positional_embedding = positional_embeddings_v1(
            edge_index
        )  # a sinusoidal encoding of j – i, representing distance along the backbone # [n_edges, 81]
        E_vectors = (
            c_alpha_coords[edge_index[0]] - c_alpha_coords[edge_index[1]]
        )  # [n_edges, 3]
        edge_s = torch.cat(
            [
                edge_connect,
                cadist,
                cedist,
                csdist,
                dis_minmax,
                _rbf(E_vectors.norm(dim=-1), D_count=16, device="cpu"),
                positional_embedding,
            ],
            dim=1,
        )
        # vector features
        orientations = get_orientations(c_alpha_coords)
        sidechains = get_sidechains(n=n_coords, ca=c_alpha_coords, c=c_coords)
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        return (
            c_alpha_coords,
            tip_coords,
            lm_embeddings,
            seq,
            node_s,
            node_v,
            edge_index,
            edge_s,
            edge_v,
        )
