##########################################################################
# File Name: sampling.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Tue 24 Oct 2023 01:47:39 PM CST
#########################################################################

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from utils.diffusion_utils import get_t_schedule, set_time, NoiseSchedule
from utils.peptide_updater import peptide_updater

def sampling(data_list, model, args, inference_steps =20,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False, actual_steps=None):
    if actual_steps is None: actual_steps = inference_steps
    N = len(data_list)
    noise_schedule = NoiseSchedule(args)
    t_schedule = get_t_schedule(inference_steps=inference_steps)
    tr_schedule, rot_schedule, tor_backbone_schedule, tor_sidechain_schedule = t_schedule, t_schedule, t_schedule, t_schedule
    for t_idx in range(actual_steps):
        t_tr, t_rot, t_tor_backbone, t_tor_sidechain = tr_schedule[t_idx], rot_schedule[t_idx], tor_backbone_schedule[t_idx], tor_sidechain_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < actual_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < actual_steps - 1 else rot_schedule[t_idx]
        dt_tor_backbone = tor_backbone_schedule[t_idx] - tor_backbone_schedule[t_idx + 1] if t_idx < actual_steps - 1 else tor_backbone_schedule[t_idx]
        dt_tor_sidechain = tor_sidechain_schedule[t_idx] - tor_sidechain_schedule[t_idx + 1] if t_idx < actual_steps - 1 else tor_sidechain_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
                # complex_graph_batch = complex_graph_batch.cuda(args.gpu)
            device = None
            tr_sigma, rot_sigma, tor_backbone_sigma, tor_sidechain_sigma = noise_schedule(t_tr, t_rot, t_tor_backbone, t_tor_sidechain)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor_backbone, t_tor_sidechain, b, device)
            
            with torch.no_grad():
                complex_graph_batch = complex_graph_batch.cuda(args.gpu)
                outputs = model(complex_graph_batch)
                tr_score, rot_score, tor_backbone_score, tor_sidechain_score = outputs.values()
            # translation gradient (?)
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(args.tr_sigma_max / args.tr_sigma_min)))
            # rotation gradient (?)
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(args.rot_sigma_max / args.rot_sigma_min)))

            # actual update
            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            # torsion gradient (?)
            tor_backbone_g = tor_backbone_sigma * torch.sqrt(torch.tensor(2 * np.log(args.tor_backbone_sigma_max / args.tor_backbone_sigma_min)))
            tor_sidechain_g = tor_sidechain_sigma * torch.sqrt(torch.tensor(2 * np.log(args.tor_sidechain_sigma_max / args.tor_sidechain_sigma_min)))
            
            if tor_backbone_score is not None:
                if ode:
                    tor_backbone_perturb = (0.5 * tor_backbone_g ** 2 * dt_tor_backbone * tor_backbone_score.cpu()).numpy()
                else:
                    tor_backbone_z = torch.zeros(tor_backbone_score.shape) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_backbone_score.shape)
                    tor_backbone_perturb = (tor_backbone_g ** 2 * dt_tor_backbone * tor_backbone_score.cpu() + tor_backbone_g * np.sqrt(dt_tor_backbone) * tor_backbone_z).numpy()
                torsions_backbone_per_molecule = tor_backbone_perturb.shape[0] // b
            else:
                torsions_backbone_per_molecule,tor_backbone_perturb = None,None
            
            if tor_sidechain_score is not None:
                if ode:
                    tor_sidechain_perturb = (0.5 * tor_sidechain_g ** 2 * dt_tor_sidechain * tor_sidechain_score.cpu()).numpy()
                else:
                    tor_sidechain_z = torch.zeros(tor_sidechain_score.shape) if no_random or (no_final_step_noise and t_idx == actual_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_sidechain_score.shape)
                    tor_sidechain_perturb = (tor_sidechain_g ** 2 * dt_tor_sidechain * tor_sidechain_score.cpu() + tor_sidechain_g * np.sqrt(dt_tor_sidechain) * tor_sidechain_z).numpy()
                torsions_sidechain_per_molecule = tor_sidechain_perturb.shape[0] // b
            else:
                torsions_sidechain_per_molecule,tor_sidechain_perturb = None,None

            # Apply noise
            new_data_list.extend([peptide_updater(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                          tor_backbone_perturb[i * torsions_backbone_per_molecule:(i + 1) * torsions_backbone_per_molecule] if torsions_backbone_per_molecule else None, tor_sidechain_perturb[i * torsions_sidechain_per_molecule:(i + 1) * torsions_sidechain_per_molecule] if torsions_sidechain_per_molecule else None)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
            # print(f'inf_{t_idx}_model: ', t1-t0)
            # print(f'inf_{t_idx}_modify: ', t2-t1)
        data_list = new_data_list

        if visualization_list is not None:
            visualization_list.append(np.asarray(
                [complex_graph['pep_a'].pos.cpu().numpy() + complex_graph.original_center.cpu().numpy() for complex_graph in data_list]))
    
    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence = []
            for complex_graph_batch in loader:
                b = complex_graph_batch.num_graphs
                set_time(complex_graph_batch, 0, 0, 0, 0, b, device)
                complex_graph_batch = complex_graph_batch.cuda(args.gpu)
                confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None
            
    return data_list, confidence, visualization_list
