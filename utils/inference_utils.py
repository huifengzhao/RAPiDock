##########################################################################
# File Name: inference_utils.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:31:15 AM CST
#########################################################################


import os
import esm
import MDAnalysis
import torch
import Bio.PDB
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
from esm import FastaBatchedDataset, pretrained
from utils.dataset_utils import three_to_one, standard_residue_sort, get_sequences
from dataset.protein_feature import get_protein_feature_mda
from dataset.peptide_feature import get_ori_peptide_feature_mda
from utils.PeptideBuilder import make_structure_from_sequence

def set_nones(l):
    return [s if str(s) != 'nan' else None for s in l]

Dihedral_angle = {
    'Helical' : [-57,-47],
    'Extended' : [-139,135],
    'Polyproline' : [-78,149]
}

def compute_ESM_embeddings(model, alphabet, labels, sequences):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 4096

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
    return embeddings

def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None

class InferenceDataset(Dataset):
    def __init__(self, output_dir, complex_name_list, protein_description_list, peptide_description_list, lm_embeddings, lm_embeddings_pep, precomputed_lm_embeddings=None, conformation_type=None, conformation_partial=None):

        super(InferenceDataset, self).__init__()

        self.output_dir = output_dir
        self.complex_names = complex_name_list
        self.protein_descriptions = protein_description_list
        self.peptide_descriptions = peptide_description_list
        self.conformation_type = conformation_type
        self.conformation_partial = conformation_partial
        
        model = None
        # generate LM embeddings for protein
        if lm_embeddings and (precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None):
            print("Generating ESM language model embeddings for protein")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = get_sequences(protein_description_list)
            labels, sequences = [], []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                sequences.extend(s)
                labels.extend([str(complex_name_list[i]) + '_chain_' + str(j) for j in range(len(s))])

            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

            self.lm_embeddings = []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                self.lm_embeddings.append([lm_embeddings[f'{complex_name_list[i]}_chain_{j}'] for j in range(len(s))])
            
        elif not lm_embeddings:
            self.lm_embeddings = [None] * len(self.complex_names)
        
        else:
            self.lm_embeddings = precomputed_lm_embeddings
        
        # generate LM embeddings for peptide
        if lm_embeddings_pep:
            print("Generating ESM language model embeddings for peptide")
            if model is None:
                model_location = "esm2_t33_650M_UR50D"
                model, alphabet = pretrained.load_model_and_alphabet(model_location)
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()

            peptide_sequences = get_sequences(peptide_description_list)
            labels, sequences = [], []
            for i in range(len(peptide_sequences)):
                s = peptide_sequences[i].split(':')
                sequences.extend(s)
                labels.extend([str(complex_name_list[i]) + '_chain_' + str(j) for j in range(len(s))])
                
            lm_embeddings_pep = compute_ESM_embeddings(model, alphabet, labels, sequences)
            
            self.lm_embeddings_pep = []
            for i in range(len(peptide_sequences)):
                s = peptide_sequences[i].split(':')
                self.lm_embeddings_pep.append([lm_embeddings_pep[f'{complex_name_list[i]}_chain_{j}'] for j in range(len(s))])
        
        else:
            self.lm_embeddings_pep = [None] * len(self.complex_names)
            
        # generate protein structures with ESMFold if only protein sequences are provided
        protein_structure_missing = len([protein_description for protein_description in protein_description_list if 'pdb' not in protein_description]) > 0
        if protein_structure_missing:
            print("generating missing protein structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()
            
            for i in range(len(protein_description_list)):
                if 'pdb' not in protein_description_list[i]:
                    self.protein_descriptions[i] = f"{output_dir}/{complex_name_list[i]}/{complex_name_list[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_descriptions[i]):
                        print("generating", self.protein_descriptions[i])
                        generate_ESM_structure(model, self.protein_descriptions[i], protein_description_list[i])
    
    def len(self):
        return len(self.complex_names)
    
    def get(self, idx):
        name, protein_file, peptide_description, lm_embedding, lm_embedding_pep = self.complex_names[idx], self.protein_descriptions[idx], self.peptide_descriptions[idx], self.lm_embeddings[idx], self.lm_embeddings_pep[idx]
        os.system(f'cp {protein_file} {self.output_dir}/{name}/{name}_protein_raw.pdb')
        # build the pytorch geometric heterogeneous graph
        c_alpha_coords_rec, tip_coords_rec, lm_embeddings_rec, seq_rec, node_s_rec, node_v_rec, edge_index_rec, edge_s_rec, edge_v_rec = get_protein_feature_mda(protein_file, lm_embedding_chains=lm_embedding)
        
        # build the initial peptide, either from file or seq
        if 'pdb' in peptide_description:
            os.system(f'cp {peptide_description} {self.output_dir}/{name}/{name}_peptide_raw.pdb')
            u = MDAnalysis.Universe(peptide_description)
            trans = {}
            seq = []
            for res_idx, residue in enumerate(u.residues):
                res_name = residue.resname.strip()
                seq.append(three_to_one[res_name] if res_name in three_to_one.keys() else f'[{res_name}]')
                trans[int(residue.resid) if residue.icode == '' else f"{residue.resid}{residue.icode.strip()}"] = res_idx
            real_idx = [trans[idx] for idx in sorted(set(trans.keys()),key=standard_residue_sort)]
            seq = [seq[i] for i in real_idx]
            seq = ''.join(seq)
            oxt = len(u.atoms.select_atoms('name OXT')) == 1
        else:
            seq = peptide_description
            oxt = True
        
        t_dict = ['Helical','Extended','Polyproline']
        partials = []
        peptide_inits = []
        if self.conformation_partial is not None:
            p_dict = {t_dict[idx]:int(_) for idx,_ in enumerate(self.conformation_partial.split(':'))}
            for t in p_dict:
                assert p_dict[t] >= 0
                if p_dict[t] != 0:
                    structure = make_structure_from_sequence(seq,phi=[Dihedral_angle[t][0]]*(len(seq)-1),psi_im1=[Dihedral_angle[t][1]]*(len(seq)-1),oxt=oxt)
                    out = Bio.PDB.PDBIO()
                    out.set_structure(structure)
                    peptide_init = os.path.join(f"{self.output_dir}/{name}",f'{name}_peptide_{t}.pdb')
                    out.save(peptide_init)
                    peptide_inits.append(peptide_init)
                    partials.append(p_dict[t])
        else:
            t = {'H':'Helical','E':'Extended','P':'Polyproline'}[self.conformation_type]
            structure = make_structure_from_sequence(seq,phi=[Dihedral_angle[t][0]]*(len(seq)-1),psi_im1=[Dihedral_angle[t][1]]*(len(seq)-1),oxt=oxt)
            out = Bio.PDB.PDBIO()
            out.set_structure(structure)
            peptide_init = os.path.join(f"{self.output_dir}/{name}",f'{name}_peptide_{t}.pdb')
            out.save(peptide_init)
            peptide_inits.append(peptide_init)
            partials.append(1)
        
        noh_mda_pep, ori_coords_pep, coords_pep, lm_embeddings_pep, seq_pep, all_edge_index_pep, backbone_edge_index_pep,sidechain_edge_index_pep, atom2res_index, atom2resid_index, atom2atomid_index, pep_a_s = get_ori_peptide_feature_mda(peptide_inits[0], match= False, lm_embedding_chains=lm_embedding_pep)
        try:
            os.remove(os.path.join(f"{self.output_dir}/{name}",'peptide_noh.pdb'))
        except:pass
        
        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph['name'] = name
        complex_graph['pep_a'].pos = coords_pep.to(dtype=torch.float)
        complex_graph['pep_a'].orig_pos = ori_coords_pep.to(dtype=torch.float)
        complex_graph['pep_a'].atom2res_index = torch.tensor(atom2res_index)
        complex_graph['pep_a'].atom2resid_index = torch.tensor(atom2resid_index)
        complex_graph['pep_a'].atom2atomid_index = torch.tensor(atom2atomid_index)
        complex_graph['pep_a'].x = pep_a_s.to(torch.int64)
        complex_graph['pep_a','pep_a'].edge_index = all_edge_index_pep
        complex_graph['pep_a','pep_a'].backbone_edge_index = backbone_edge_index_pep
        complex_graph['pep_a','pep_a'].sidechain_edge_index = sidechain_edge_index_pep

        G = to_networkx(complex_graph.to_homogeneous(), to_undirected=False)
        edges = complex_graph['pep_a','pep_a'].edge_index.T.numpy()
        backbone_edges = complex_graph['pep_a','pep_a'].backbone_edge_index.T.tolist()
        sidechain_edges = complex_graph['pep_a','pep_a'].sidechain_edge_index.T.tolist()
        ## sidechain
        to_rotate_sidechain = []
        for i in range(0, edges.shape[0], 2):
            assert edges[i, 0] == edges[i+1, 1]
            G2 = G.to_undirected()
            G2.remove_edge(*edges[i])
            if not nx.is_connected(G2) and edges[i].tolist() in sidechain_edges:
                l = list(sorted(nx.connected_components(G2), key=len)[0])
                if len(l) > 1:
                    if edges[i, 0] in l:
                        to_rotate_sidechain.append([])
                        to_rotate_sidechain.append(l)
                    else:
                        to_rotate_sidechain.append(l)
                        to_rotate_sidechain.append([])
                    continue
            to_rotate_sidechain.append([])
            to_rotate_sidechain.append([])
        mask_edges_sidechain = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate_sidechain], dtype=bool)
        mask_rotate_sidechain = np.zeros((np.sum(mask_edges_sidechain), len(G.nodes())), dtype=bool)
        idx = 0
        for i in range(len(G.edges())):
            if mask_edges_sidechain[i]:
                mask_rotate_sidechain[idx][np.asarray(to_rotate_sidechain[i], dtype=int)] = True
                idx += 1
                
        ## backbone
        to_rotate_backbone = []
        for i in range(0, edges.shape[0], 2):
            assert edges[i, 0] == edges[i+1, 1]

            G2 = G.to_undirected()
            G2.remove_edge(*edges[i])
            if not nx.is_connected(G2) and edges[i].tolist() in backbone_edges:
                l = list(sorted(nx.connected_components(G2), key=len)[0])
                if len(l) > 1:
                    if edges[i, 0] in l:
                        to_rotate_backbone.append([])
                        to_rotate_backbone.append(l)
                    else:
                        to_rotate_backbone.append(l)
                        to_rotate_backbone.append([])
                    continue
            to_rotate_backbone.append([])
            to_rotate_backbone.append([])
        mask_edges_backbone = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate_backbone], dtype=bool)
        mask_rotate_backbone = np.zeros((np.sum(mask_edges_backbone), len(G.nodes())), dtype=bool)
        idx = 0
        for i in range(len(G.edges())):
            if mask_edges_backbone[i]:
                mask_rotate_backbone[idx][np.asarray(to_rotate_backbone[i], dtype=int)] = True
                idx += 1
                
        complex_graph['pep_a'].mask_edges_sidechain = torch.from_numpy(mask_edges_sidechain)
        complex_graph['pep_a'].mask_rotate_sidechain = mask_rotate_sidechain
        complex_graph['pep_a'].mask_edges_backbone = torch.from_numpy(mask_edges_backbone)
        complex_graph['pep_a'].mask_rotate_backbone = mask_rotate_backbone
        num_residues = len(c_alpha_coords_rec)
        if num_residues <= 1:
            raise ValueError(f"rec contains only 1 residue!")
        
        complex_graph['receptor'].x = torch.cat([seq_rec.reshape([-1,1]),node_s_rec, torch.tensor(lm_embeddings_rec)], axis=1) if lm_embeddings_rec is not None else torch.cat([seq_rec.reshape([-1,1]),node_s_rec], axis=1) # [num_res, 1+9+1280]
        complex_graph['receptor'].pos = c_alpha_coords_rec.to(dtype=torch.float)
        complex_graph['receptor'].tips = tip_coords_rec.to(dtype=torch.float)
        complex_graph['receptor'].node_v = node_v_rec.to(dtype=torch.float)
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = edge_index_rec
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_s = edge_s_rec.to(dtype=torch.float)
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_v = edge_v_rec.to(dtype=torch.float)
        complex_graph['pep'].x = torch.cat([seq_pep.reshape([-1,1]), torch.tensor(lm_embeddings_pep)], axis=1) if lm_embeddings_pep is not None else seq_pep.reshape([-1,1])
        complex_graph['pep'].noh_mda = noh_mda_pep
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True).to(dtype=torch.float)
        complex_graph['receptor'].pos -= protein_center
        complex_graph['receptor'].tips -= protein_center
        complex_graph['pep_a'].pos -= protein_center
        complex_graph.original_center = protein_center
        complex_graph['success'] = True
        complex_graph['partials'] = partials
        complex_graph['peptide_inits'] = peptide_inits
        return complex_graph
