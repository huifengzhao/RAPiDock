##########################################################################
# File Name: diffusion_utils.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Tue 24 Oct 2023 01:47:39 PM CST
#########################################################################

import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F

class NoiseSchedule:
    """
        Transforms t into scaled sigmas
    """
    def __init__(self, args):
        self.tr_sigma_min = args.tr_sigma_min # 0.1
        self.tr_sigma_max = args.tr_sigma_max # 
        self.rot_sigma_min = args.rot_sigma_min # 0.1
        self.rot_sigma_max = args.rot_sigma_max # 1.65
        self.tor_backbone_sigma_min = args.tor_backbone_sigma_min # 0.0314
        self.tor_backbone_sigma_max = args.tor_backbone_sigma_max # 3.14 
        self.tor_sidechain_sigma_min = args.tor_sidechain_sigma_min # 0.0314
        self.tor_sidechain_sigma_max = args.tor_sidechain_sigma_max # 3.14

    def __call__(self, t_tr, t_rot, t_tor_backbone, t_tor_sidechain):
        """
            Convert from time to (scaled) sigma space
        """
        tr_sigma = self.tr_sigma_min ** (1 - t_tr) * self.tr_sigma_max**t_tr
        rot_sigma = self.rot_sigma_min ** (1 - t_rot) * self.rot_sigma_max**t_rot
        tor_backbone_sigma = self.tor_backbone_sigma_min ** (1 - t_tor_backbone) * self.tor_backbone_sigma_max**t_tor_backbone
        tor_sidechain_sigma = self.tor_sidechain_sigma_min ** (1 - t_tor_sidechain) * self.tor_sidechain_sigma_max**t_tor_sidechain
        return tr_sigma, rot_sigma, tor_backbone_sigma, tor_sidechain_sigma
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_size, scale):
        super().__init__()
        self.embed_dim = embedding_size
        self.scale = scale
        self.max_positions = 1e4

    def forward(self, x):
        """
        From https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        """
        assert len(x.shape) == 1
        x = self.scale * x

        half_dim = self.embed_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim,
                dtype=torch.float32,
                device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (x.shape[0], self.embed_dim)
        return emb
    
class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size // 2) * scale,
            requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)
        return emb

def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]

def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == "sinusoidal":
        emb_func = SinusoidalEmbedding(embedding_size = embedding_dim, scale = embedding_scale)
    elif embedding_type == "fourier":
        emb_func = GaussianFourierProjection(embedding_size = embedding_dim, scale = embedding_scale)
    else:
        raise NotImplemented
    return emb_func

def set_time(complex_graphs, t_tr, t_rot, t_tor_backbone, t_tor_sidechain, batch_size: int, device=None):
    """
        Save sampled time to current batch
    """
    pep_size = complex_graphs["pep"].num_nodes
    complex_graphs["pep"].node_t = {
        "tr": t_tr * torch.ones(pep_size).to(device),
        "rot": t_rot * torch.ones(pep_size).to(device),
        "tor_backbone": t_tor_backbone * torch.ones(pep_size).to(device),
        "tor_sidechain": t_tor_sidechain * torch.ones(pep_size).to(device),
    }
    rec_size = complex_graphs["receptor"].num_nodes
    complex_graphs["receptor"].node_t = {
        "tr": t_tr * torch.ones(rec_size).to(device),
        "rot": t_rot * torch.ones(rec_size).to(device),
        "tor_backbone": t_tor_backbone * torch.ones(rec_size).to(device),
        "tor_sidechain": t_tor_sidechain * torch.ones(rec_size).to(device),
    }
    pep_atom_size = complex_graphs["pep_a"].num_nodes
    complex_graphs["pep_a"].node_t = {
        "tr": t_tr * torch.ones(pep_atom_size).to(device),
        "rot": t_rot * torch.ones(pep_atom_size).to(device),
        "tor_backbone": t_tor_backbone * torch.ones(pep_atom_size).to(device),
        "tor_sidechain": t_tor_sidechain * torch.ones(pep_atom_size).to(device),
    }
    complex_graphs.complex_t = {
        "tr": t_tr * torch.ones(batch_size).to(device),
        "rot": t_rot * torch.ones(batch_size).to(device),
        "tor_backbone": t_tor_backbone * torch.ones(batch_size).to(device),
        "tor_sidechain": t_tor_sidechain * torch.ones(batch_size).to(device),
    }
