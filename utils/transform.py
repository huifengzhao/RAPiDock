##########################################################################
# File Name: transform.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Fri 20 Oct 2023 11:09:44 AM CST
#########################################################################

import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from utils.so3 import sample_vec, score_vec
from utils.torus import score as score_torus
from utils.peptide_updater import peptide_updater
from utils.diffusion_utils import set_time, NoiseSchedule

class NoiseTransform(BaseTransform):
    """
        Apply translation, rotation, torsional noise
    """
    def __init__(self, args):
        # save min/max sigma scales
        self.noise_schedule = NoiseSchedule(args)
        
    def __call__(self, data):
        """
            Modifies data in place
            @param (torch_geometric.data.HeteroData) data
        """
        t = np.random.uniform()  # sample time
        t_tr, t_rot, t_tor_backbone, t_tor_sidechain = t, t, t, t  # same time scale for each
        data = self.apply_noise(data, t_tr, t_rot, t_tor_backbone, t_tor_sidechain)
        return data
    
    def apply_noise(self, data, t_tr, t_rot, t_tor_backbone, t_tor_sidechain,
                    tr_update=None,
                    rot_update=None,
                    tor_backbone_updates=None,
                    tor_sidechain_updates=None,):
        """
            Apply noise to existing HeteroData object
            @param (torch_geometric.data.HeteroData) data
        """
        tr_sigma, rot_sigma, tor_backbone_sigma, tor_sidechain_sigma = self.noise_schedule(t_tr, t_rot, t_tor_backbone, t_tor_sidechain)
        set_time(data, t_tr, t_rot, t_tor_backbone, t_tor_sidechain, 1, device=None)
        # sample updates if not provided
        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        tor_backbone_updates = np.random.normal(loc=0.0, scale=tor_backbone_sigma, size=data['pep_a'].mask_edges_backbone.sum()) if tor_backbone_updates is None else tor_backbone_updates
        tor_sidechain_updates = np.random.normal(loc=0.0, scale=tor_sidechain_sigma, size=data['pep_a'].mask_edges_sidechain.sum()) if tor_sidechain_updates is None else tor_sidechain_updates
        
        # apply updates
        peptide_updater(data, tr_update, torch.from_numpy(rot_update).float(),tor_backbone_updates, tor_sidechain_updates)
        # compute ground truth score given updates, noise level
        get_score(data, tr_update, tr_sigma, rot_update, rot_sigma, tor_backbone_updates, tor_backbone_sigma, tor_sidechain_updates, tor_sidechain_sigma)
        
        return data
    
def get_score(data, tr_update, tr_sigma, rot_update, rot_sigma, tor_backbone_updates, tor_backbone_sigma, tor_sidechain_updates, tor_sidechain_sigma):
    """
        Compute ground truth score, given updates and noise.
        Modifies data in place.
    """
    # translation score
    data.tr_score = -tr_update / tr_sigma**2
    # rotation score
    rot_score = torch.from_numpy(score_vec(vec=rot_update, eps=rot_sigma)).float()
    rot_score = rot_score.unsqueeze(0)
    data.rot_score = rot_score
    # torsion score
    tor_backbone_score = score_torus(tor_backbone_updates, tor_backbone_sigma)
    tor_sidechain_score = score_torus(tor_sidechain_updates, tor_sidechain_sigma)
    data.tor_backbone_score = torch.from_numpy(tor_backbone_score).float()
    data.tor_sidechain_score = torch.from_numpy(tor_sidechain_score).float()
    
    tor_backbone_s_edge = np.ones(data["pep_a"].mask_edges_backbone.sum())
    data.tor_backbone_s_edge = tor_backbone_s_edge * tor_backbone_sigma
    tor_sidechain_s_edge = np.ones(data["pep_a"].mask_edges_sidechain.sum())
    data.tor_sidechain_s_edge = tor_sidechain_s_edge * tor_sidechain_sigma
    
    return data
