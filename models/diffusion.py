##########################################################################
# File Name: diffusion.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Fri 20 Oct 2023 01:09:20 PM CST
#########################################################################

import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_cluster import radius
from torch_scatter import scatter, scatter_mean
from e3nn import o3
from e3nn.nn import BatchNorm
from utils.diffusion_utils import get_timestep_embedding
from utils.transform import NoiseSchedule
from utils.so3 import score_norm
from utils.torus import score_norm as torus_score
from dataset.peptide_feature import three2idx, allowable_features, atomname2idx, get_updated_peptide_feature

feature_dims = [
            len(three2idx), max([len(res) for res in atomname2idx]) # Amino_idx_dim/ Atom_idx_dim
        ] + \
        [
            len(value) for key, value in allowable_features.items() # Atom_features_dim
        ] + [4] # Atom_charity_center_dim

class GaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class AminoEmbedding(nn.Module):
    """
        Embeddings for atom identity only (as of now)
    """
    def __init__(self, num_amino, emb_dim, sigma_embed_dim, intra_dim, dihediral_dim, lm_embedding_type = None):
        super(AminoEmbedding, self).__init__()
        self.amino_type_dim = 1
        self.sigma_embed_dim = sigma_embed_dim
        self.intra_dim = intra_dim
        self.dihediral_dim = dihediral_dim
        self.lm_embedding_type = lm_embedding_type
        
        self.amino_ebd = nn.Embedding(num_amino+1, emb_dim) # add 1 for padding
        self.sigma_ebd = nn.Linear(sigma_embed_dim, emb_dim)
        self.intra_dis_ebd = nn.Linear(intra_dim, emb_dim)
        self.dihediral_angles_ebd = nn.Linear(dihediral_dim, emb_dim)
        
        # LM embedding (ESM2)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
                self.lm_embedding_layer = nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)

    def forward(self, x):
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.amino_type_dim + self.sigma_embed_dim + self.intra_dim + self.dihediral_dim + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.amino_type_dim + self.sigma_embed_dim + self.intra_dim + self.dihediral_dim
        
        x_embedding = 0
        # amino sequence
        x_embedding += self.amino_ebd(x[:,:self.amino_type_dim].long()).squeeze() # amino_ebd
        # sigma noise embedding
        x_embedding += self.sigma_ebd(x[:,-self.sigma_embed_dim:].to(torch.float32)) # sigma_ebd
        x_embedding += self.intra_dis_ebd(x[:,self.amino_type_dim:self.amino_type_dim+self.intra_dim].to(torch.float32)) # intra_dis_ebd
        x_embedding += self.dihediral_angles_ebd(x[:,self.amino_type_dim+self.intra_dim:self.amino_type_dim+self.intra_dim+self.dihediral_dim].to(torch.float32)) # dihediral_angles_ebd
        # # consider LM embedding here
        if self.lm_embedding_type is not None:
            x_embedding =  self.lm_embedding_layer(torch.cat([x_embedding, x[:, self.amino_type_dim+self.intra_dim+self.dihediral_dim:self.amino_type_dim+self.intra_dim+self.dihediral_dim+self.lm_embedding_dim].to(torch.float32)], dim=1))
        return x_embedding

class AtomEncoder(nn.Module):

    def __init__(self, emb_dim, sigma_embed_dim, pep_attr_dim):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        self.num_scalar_features = sigma_embed_dim
        self.emb_dim = emb_dim
        self.pep_attr_dim = pep_attr_dim
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        self.final_layer = torch.nn.Linear(emb_dim + pep_attr_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.pep_attr_dim
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        x_embedding = self.final_layer(torch.cat([x_embedding, x[:, -self.pep_attr_dim:]], axis=1))
        return x_embedding
    
class EdgeEmbedding(nn.Module):
    """
        Embeddings for edge feature (as of now)
    """
    def __init__(self, ns,sigma_embed_dim=32,feature_embed_dim=103, dropout=0.0):
        self.connect_dim = 1
        self.sigma_embed_dim = sigma_embed_dim
        self.feature_embed_dim = feature_embed_dim
        
        super(EdgeEmbedding, self).__init__()
        self.connect_ebd = nn.Embedding(2, ns) # 0,1
        self.sigma_ebd = nn.Linear(self.sigma_embed_dim, ns)
        self.feature_ebd = nn.Linear(self.feature_embed_dim, ns)
        self.edge_embedding = nn.Sequential(nn.Linear(ns*3, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        # _init(self)

    def forward(self, x):
        # connect
        connect_ebd = self.connect_ebd(x[:,self.sigma_embed_dim:self.sigma_embed_dim+self.connect_dim].long()).squeeze()
        # sigma noise embedding
        sigma_ebd = self.sigma_ebd(x[:,:self.sigma_embed_dim].to(torch.float32))
        # 
        feature_ebd = self.feature_ebd(x[:,-self.feature_embed_dim:].to(torch.float32))
        # # add together
        x_embedding = torch.cat([connect_ebd, sigma_ebd, feature_ebd], 1)
        
        x_embedding = self.edge_embedding(x_embedding)
        return x_embedding
    
class CGTPEL(nn.Module):
    """
        Clebsch-Gordan tensor product equivariant layer
    """
    def __init__(self, 
                 in_irreps, sh_irreps, out_irreps, n_edge_features,
                 residual=True, batch_norm=True,hidden_features=None, is_last_layer=False,dropout=0.0):
        super(CGTPEL, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.tensor_prod.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if (batch_norm and not is_last_layer) else None

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        out_nodes=None,
        reduction="mean",
    ):
        """
        @param edge_index  [2, E]
        @param edge_sh  edge spherical harmonics
        """
        edge_src, edge_dst = edge_index
        tp = self.tensor_prod(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduction)

        if self.residual:
            new_shape = (0, out.shape[-1] - node_attr.shape[-1])
            padded = F.pad(node_attr, new_shape)
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out
    
class CGTensorProductEquivariantModel(nn.Module):
    """
        Clebsch-Gordan tensor product-based equivariant model
    """
    def __init__(self, args,
            confidence_mode=False,
            confidence_dropout=0,
            confidence_no_batchnorm=False,
            num_confidence_outputs=1,
        ):
        super(CGTensorProductEquivariantModel, self).__init__()
        
        self.dropout = args.dropout
        self.cross_cutoff_weight = args.cross_cutoff_weight
        self.cross_cutoff_bias = args.cross_cutoff_bias
        self.cross_max_dist = args.cross_max_distance
        self.dynamic_max_cross=args.dynamic_max_cross
        self.center_max_dist = args.center_max_distance
        
        self.ns, self.nv = args.ns, args.nv
        ns, nv = self.ns, self.nv
        self.num_conv_layers = args.num_conv_layers
        self.lig_max_radius = args.max_radius
        self.batch_norm = not args.no_batch_norm
        self.confidence_mode = confidence_mode
        self.scale_by_sigma = args.scale_by_sigma
        self.rec_amino_dim = args.rec_amino_dim
        self.pep_amino_dim = args.pep_amino_dim 
        self.sigma_embed_dim = args.sigma_embed_dim
        self.intra_dim = args.intra_dim
        self.dihediral_dim = args.dihediral_dim
        self.edge_feature_dim = args.edge_feature_dim
        self.embedding_type = args.embedding_type
        self.embedding_scale = args.embedding_scale
        self.cross_dist_embed_dim = args.cross_distance_embed_dim
        self.use_second_order_repr = args.use_second_order_repr
        self.dist_embed_dim = args.distance_embed_dim
        self.esm_embeddings_peptide = args.esm_embeddings_peptide_train is not None
        self.noise_schedule = NoiseSchedule(args) if not confidence_mode else None
        self.timestep_emb_func = get_timestep_embedding(self.embedding_type,self.sigma_embed_dim,self.embedding_scale)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)
        self.rec_node_embedding = AminoEmbedding(self.rec_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, self.dihediral_dim, 'esm')
        self.rec_edge_embedding = EdgeEmbedding(ns,self.sigma_embed_dim, self.edge_feature_dim, self.dropout)
        self.pep_node_embedding = AminoEmbedding(self.pep_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, 3, 'esm') if self.esm_embeddings_peptide else AminoEmbedding(self.pep_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, 3)
        self.pep_edge_embedding = EdgeEmbedding(ns,self.sigma_embed_dim, self.edge_feature_dim, self.dropout)
        self.top_k = args.top_k
        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_dist_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(ns, ns),
        )
        self.cross_distance_expansion = GaussianSmearing(
                0.0, self.cross_max_dist, self.cross_dist_embed_dim)
        
        if self.use_second_order_repr:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]
        
        intra_convs = []
        cross_convs = []
        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            params = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 3 * ns,
                "hidden_features": 3 * ns,
                "residual": False,
                'batch_norm': self.batch_norm,
                'dropout': self.dropout
            }
            intra_convs.append(CGTPEL(**params))
            cross_convs.append(CGTPEL(**params))
            
        self.intra_convs = nn.ModuleList(intra_convs)
        self.cross_convs = nn.ModuleList(cross_convs)
        
        # compute confidence score
        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs),
            )
        else:
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(
                0.0, self.center_max_dist, self.dist_embed_dim
            )
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(self.dist_embed_dim + self.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(ns, ns)
            )
            
            self.final_conv = CGTPEL(
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=self.dropout,
                batch_norm= self.batch_norm,
                is_last_layer=True,
            )
            
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + self.sigma_embed_dim, ns),nn.Dropout(self.dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + self.sigma_embed_dim, ns),nn.Dropout(self.dropout), nn.ReLU(), nn.Linear(ns, 1))
            
            # Build Pep_atom_ebd
            self.pep_a_node_embedding = AtomEncoder(o3.Irreps(self.intra_convs[-1].out_irreps).dim,self.sigma_embed_dim, o3.Irreps(self.intra_convs[-1].out_irreps).dim)
            
            self.lig_distance_expansion = GaussianSmearing(
                0.0, self.lig_max_radius, self.dist_embed_dim)
            
            # torsion angles components
            self.final_edge_embedding = nn.Sequential(
                    nn.Linear(self.dist_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, ns),
                )
            self.final_tp_tor_bb = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.final_tp_tor_sc = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bb_bond_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor_bb.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            self.tor_sc_bond_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor_sc.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            self.tor_bb_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            self.tor_sc_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            
    def forward(self, _data):
        data = copy.copy(_data)
        # get noise schedule
        tr_t = data.complex_t["tr"]
        rot_t = data.complex_t["rot"]
        tor_backbone_t = data.complex_t["tor_backbone"]
        tor_sidechain_t = data.complex_t["tor_sidechain"]
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_backbone_sigma,tor_sidechain_sigma = self.noise_schedule(tr_t, rot_t, tor_backbone_t,tor_sidechain_t)
        else:
            tr_sigma, rot_sigma, tor_backbone_sigma,tor_sidechain_sigma = tr_t, rot_t, tor_backbone_t,tor_sidechain_t
            
        self.device = data['pep'].x.device

        # build receptor graph
        receptor_graph = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(receptor_graph[0])
        rec_src, rec_dst = rec_edge_index = receptor_graph[1]
        rec_edge_attr = self.rec_edge_embedding(receptor_graph[2])
        rec_edge_sh = receptor_graph[3]
        
        # build pep graph
        node_s_pep, ca_pep, tips_pep, edge_index_pep, edge_s_pep, edge_v_pep = get_updated_peptide_feature(data, self.device, self.top_k)
        data['pep'].x = torch.cat([data['pep'].x[:,:1], node_s_pep, data['pep'].x[:,-1280:]], axis=1) if self.esm_embeddings_peptide else torch.cat([data['pep'].x[:,:1], node_s_pep], axis=1)  # [num_res, 1+5+3(+1280)]
        data['pep'].pos = ca_pep.to(dtype=torch.float)
        data['pep'].tips = tips_pep.to(dtype=torch.float)
        data['pep', 'pep_contact', 'pep'].edge_index = edge_index_pep
        data['pep', 'pep_contact', 'pep'].edge_s = edge_s_pep.to(dtype=torch.float)
        data['pep', 'pep_contact', 'pep'].edge_v = edge_v_pep.to(dtype=torch.float)
        
        pep_graph = self.build_pep_conv_graph(data)
        pep_node_attr = self.pep_node_embedding(pep_graph[0])
        pep_src, pep_dst = pep_edge_index = pep_graph[1]
        pep_edge_attr = self.pep_edge_embedding(pep_graph[2])
        pep_edge_sh = pep_graph[3]
        
        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * self.cross_cutoff_weight + self.cross_cutoff_bias).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(
            data, cross_cutoff
        )
        cross_pep, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)
        
        for idx in range(len(self.intra_convs)):
            # message passing within pep graph (intra)
            pep_edge_attr_ = torch.cat([pep_edge_attr, pep_node_attr[pep_src, :self.ns], pep_node_attr[pep_dst, :self.ns]], -1)
            pep_intra_update = self.intra_convs[idx](pep_node_attr, pep_edge_index, pep_edge_attr_, pep_edge_sh)
            
            # message passing between two graphs (inter)
            rec_to_pep_edge_attr_ = torch.cat([cross_edge_attr, pep_node_attr[cross_pep, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            pep_inter_update = self.cross_convs[idx](rec_node_attr, cross_edge_index, rec_to_pep_edge_attr_, cross_edge_sh, out_nodes=pep_node_attr.shape[0])
            
            # message passing within receptor graph (intra)
            if idx != len(self.intra_convs) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.intra_convs[idx](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
                
                pep_to_rec_edge_attr_ = torch.cat([cross_edge_attr, pep_node_attr[cross_pep, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.cross_convs[idx](pep_node_attr, torch.flip(cross_edge_index, dims=[0]), pep_to_rec_edge_attr_, cross_edge_sh, out_nodes=rec_node_attr.shape[0])
            
            # padding original features
            pep_node_attr = F.pad(
                pep_node_attr,
                (0, pep_intra_update.shape[-1] - pep_node_attr.shape[-1]))
            # update features with residual updates
            pep_node_attr = pep_node_attr + pep_intra_update + pep_inter_update
            
            if idx != len(self.intra_convs) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update
                
        # compute confidence score
        if self.confidence_mode:
            scalar_pep_attr = (
                torch.cat(
                    [pep_node_attr[:, : self.ns], pep_node_attr[:, -self.ns :]], dim=1
                )
                if self.num_conv_layers >= 3
                else pep_node_attr[:, : self.ns]
            )
            confidence = self.confidence_predictor(scatter_mean(scalar_pep_attr, data['pep'].batch, dim=0)).squeeze(dim=-1)
            return confidence
        
        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, pep_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(pep_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
        
        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])
        
        # fix the magnitude of tr and rot score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))
        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['pep'].x.device)

        # torsional components
        
        pep_a_node_attr_from_pep = torch.cat([pep_node_attr[data['pep'].batch == i][data['pep_a'].atom2res_index[data['pep_a'].batch == i]] for i in range(len(data.name))],dim=0)
        data['pep_a'].node_sigma_emb = self.timestep_emb_func(data['pep_a'].node_t['tr']) # tr rot and tor noise is all the same
        pep_a_node_attr = torch.cat(
                [data['pep_a'].atom2resid_index.unsqueeze(-1), data['pep_a'].atom2atomid_index.unsqueeze(-1), data['pep_a'].x, data['pep_a'].node_sigma_emb,pep_a_node_attr_from_pep], 1
            ) # 1+1+19+32+??84
        pep_a_node_attr = self.pep_a_node_embedding(pep_a_node_attr)
        
        
        if not data['pep_a'].mask_edges_backbone.squeeze().sum() == 0:
            tor_bonds_backbone, tor_edge_index_backbone, tor_edge_attr_backbone, tor_edge_sh_backbone = self.build_backbone_bond_conv_graph(data)
            tor_bond_vec_backbone = data['pep_a'].pos[tor_bonds_backbone[1]] - data['pep_a'].pos[tor_bonds_backbone[0]]
            tor_bond_attr_backbone = pep_a_node_attr[tor_bonds_backbone[0]] + pep_a_node_attr[tor_bonds_backbone[1]]
            tor_bonds_sh_backbone = o3.spherical_harmonics(
                "2e", tor_bond_vec_backbone, normalize=True, normalization="component"
            )
            tor_edge_sh_backbone = self.final_tp_tor_bb(tor_edge_sh_backbone, tor_bonds_sh_backbone[tor_edge_index_backbone[0]])
            tor_edge_attr_backbone = torch.cat(
                [
                    tor_edge_attr_backbone,
                    pep_a_node_attr[tor_edge_index_backbone[1], : self.ns],
                    tor_bond_attr_backbone[tor_edge_index_backbone[0], : self.ns],
                ],
                -1,
            )
            tor_pred_backbone = self.tor_bb_bond_conv(
                pep_a_node_attr,
                tor_edge_index_backbone,
                tor_edge_attr_backbone,
                tor_edge_sh_backbone,
                out_nodes=data['pep_a'].mask_edges_backbone.sum(),
                reduction="mean",
            )
            tor_pred_backbone = self.tor_bb_final_layer(tor_pred_backbone).squeeze(1)
            edge_sigma_backbone = tor_backbone_sigma[data["pep_a"].batch][
                data["pep_a", "pep_a"].edge_index[0]
            ][data['pep_a'].mask_edges_backbone.squeeze()]
            if self.scale_by_sigma:
                tor_pred_backbone = tor_pred_backbone * torch.sqrt(
                    torch.tensor(torus_score(edge_sigma_backbone.cpu().numpy()))
                    .float()
                    .to(data["pep_a"].x.device)
                )
        else: tor_pred_backbone = torch.empty(0, device=self.device)
        
        if not data['pep_a'].mask_edges_sidechain.squeeze().sum() == 0:
            tor_bonds_sidechain, tor_edge_index_sidechain, tor_edge_attr_sidechain, tor_edge_sh_sidechain = self.build_sidechain_bond_conv_graph(data)
            tor_bond_vec_sidechain = data['pep_a'].pos[tor_bonds_sidechain[1]] - data['pep_a'].pos[tor_bonds_sidechain[0]]
            tor_bond_attr_sidechain = pep_a_node_attr[tor_bonds_sidechain[0]] + pep_a_node_attr[tor_bonds_sidechain[1]]
            tor_bonds_sh_sidechain = o3.spherical_harmonics(
                "2e", tor_bond_vec_sidechain, normalize=True, normalization="component"
            )
            tor_edge_sh_sidechain = self.final_tp_tor_sc(tor_edge_sh_sidechain, tor_bonds_sh_sidechain[tor_edge_index_sidechain[0]])
            tor_edge_attr_sidechain = torch.cat(
                [
                    tor_edge_attr_sidechain,
                    pep_a_node_attr[tor_edge_index_sidechain[1], : self.ns],
                    tor_bond_attr_sidechain[tor_edge_index_sidechain[0], : self.ns],
                ],
                -1,
            )
            tor_pred_sidechain = self.tor_sc_bond_conv(
                pep_a_node_attr,
                tor_edge_index_sidechain,
                tor_edge_attr_sidechain,
                tor_edge_sh_sidechain,
                out_nodes=data['pep_a'].mask_edges_sidechain.sum(),
                reduction="mean",
            )
            tor_pred_sidechain = self.tor_sc_final_layer(tor_pred_sidechain).squeeze(1)
            edge_sigma_sidechain = tor_sidechain_sigma[data["pep_a"].batch][
                data["pep_a", "pep_a"].edge_index[0]
            ][data['pep_a'].mask_edges_sidechain.squeeze()]
            if self.scale_by_sigma:
                tor_pred_sidechain = tor_pred_sidechain * torch.sqrt(
                    torch.tensor(torus_score(edge_sigma_sidechain.cpu().numpy()))
                    .float()
                    .to(data["pep_a"].x.device)
                )
        else: tor_pred_sidechain = torch.empty(0, device=self.device)
        return tr_pred, rot_pred, tor_pred_backbone, tor_pred_sidechain
            
    def build_rec_conv_graph(self,data):
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        if len(data['receptor'].x.shape) == 1:
            data['receptor'].x = data['receptor'].x[:,None]
        node_attr = torch.cat(
                [data['receptor'].x, data['receptor'].node_sigma_emb], 1
            ) # 1+5+4+1280+32
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        edge_vec = data['receptor', 'receptor'].edge_v.squeeze()
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, data['receptor', 'receptor'].edge_s], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_pep_conv_graph(self,data):
        data['pep'].node_sigma_emb = self.timestep_emb_func(data['pep'].node_t['tr']) # tr rot and tor noise is all the same
        if len(data['pep'].x.shape) == 1:
            data['pep'].x = data['pep'].x[:,None]
        node_attr = torch.cat(
                [data['pep'].x, data['pep'].node_sigma_emb], 1
            ) # 1+5+4+32
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['pep', 'pep'].edge_index
        edge_vec = data['pep', 'pep'].edge_v.squeeze()
        edge_sigma_emb = data['pep'].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, data['pep', 'pep'].edge_s], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between pep and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index_ca_ca = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].pos / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
            edge_index_tips_ca = radius(data['receptor'].tips / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].pos / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
            edge_index_ca_tips = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].tips / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
            edge_index_tips_tips = radius(data['receptor'].tips / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].tips / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
        else:
            edge_index_ca_ca = radius(data['receptor'].pos, data['pep'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
            edge_index_tips_ca = radius(data['receptor'].tips, data['pep'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
            edge_index_ca_tips = radius(data['receptor'].pos, data['pep'].tips, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
            edge_index_tips_tips = radius(data['receptor'].tips, data['pep'].tips, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=10000)
        
        edge_index = torch.cat((edge_index_ca_ca, edge_index_tips_ca, edge_index_ca_tips, edge_index_tips_tips), dim=1)
        edge_index = torch.unique(edge_index,dim=1).reshape(2,-1)
        
        src, dst = edge_index
        edge_vec_ca_ca = data['receptor'].pos[dst.long()] - data['pep'].pos[src.long()]
        edge_vec_tips_ca = data['receptor'].tips[dst.long()] - data['pep'].pos[src.long()]
        edge_vec_ca_tips = data['receptor'].pos[dst.long()] - data['pep'].tips[src.long()]
        edge_vec_tips_tips = data['receptor'].tips[dst.long()] - data['pep'].tips[src.long()]
        
        edge_vec_min, indices = torch.cat((edge_vec_ca_ca.norm(dim=-1)[:,None],edge_vec_tips_ca.norm(dim=-1)[:,None],edge_vec_ca_tips.norm(dim=-1)[:,None],edge_vec_tips_tips.norm(dim=-1)[:,None]),dim=1).min(dim=1)

        edge_length_emb = self.cross_distance_expansion(edge_vec_min)
        # cross_edge_type_emb = self.cross_edge_type(indices)
        edge_sigma_emb = data['pep'].node_sigma_emb[src.long()]
        # edge_attr = torch.cat([cross_edge_type_emb, edge_sigma_emb, edge_length_emb], 1)
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec_ca_ca, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        edge_index = torch.cat([data['pep'].batch.unsqueeze(0), torch.arange(len(data['pep'].batch)).to(data['pep'].x.device).unsqueeze(0)], dim=0)
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['pep'].x.device), torch.zeros((data.num_graphs, 3)).to(data['pep'].x.device)
        center_pos.index_add_(0, index=data['pep'].batch, source=data['pep'].pos)
        center_pos = center_pos / torch.bincount(data['pep'].batch).unsqueeze(1)

        edge_vec = data['pep'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['pep'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    def build_backbone_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['pep_a', 'pep_a'].edge_index[:, data['pep_a'].mask_edges_backbone.squeeze()].long()
        bond_pos = (data['pep_a'].pos[bonds[0]] + data['pep_a'].pos[bonds[1]]) / 2
        bond_batch = data['pep_a'].batch[bonds[0]]
        edge_index = radius(data['pep_a'].pos, bond_pos, self.lig_max_radius, batch_x=data['pep_a'].batch, batch_y=bond_batch)

        edge_vec = data['pep_a'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
    
    def build_sidechain_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['pep_a', 'pep_a'].edge_index[:, data['pep_a'].mask_edges_sidechain.squeeze()].long()
        bond_pos = (data['pep_a'].pos[bonds[0]] + data['pep_a'].pos[bonds[1]]) / 2
        bond_batch = data['pep_a'].batch[bonds[0]]
        edge_index = radius(data['pep_a'].pos, bond_pos, self.lig_max_radius, batch_x=data['pep_a'].batch, batch_y=bond_batch)

        edge_vec = data['pep_a'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
