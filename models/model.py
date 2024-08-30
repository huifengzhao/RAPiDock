##########################################################################
# File Name: model.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Fri 20 Oct 2023 01:07:48 PM CST
#########################################################################

import torch.nn as nn
from .diffusion import CGTensorProductEquivariantModel

class BaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, 
                 confidence_mode=False,num_confidence_outputs=1):
        super(BaseModel, self).__init__()

        # raw encoders
        self.encoder = CGTensorProductEquivariantModel(args, confidence_mode=confidence_mode, num_confidence_outputs = num_confidence_outputs)

class ScoreModel(BaseModel):
    def __init__(self, args):
        super(ScoreModel, self).__init__(args)

    def forward(self, batch):
        # move graphs to cuda
        tr_pred, rot_pred, tor_pred_backbone, tor_pred_sidechain = self.encoder(batch)
        outputs = {}
        outputs["tr_pred"] = tr_pred
        outputs["rot_pred"] = rot_pred
        outputs["tor_pred_backbone"] = tor_pred_backbone
        outputs["tor_pred_sidechain"] = tor_pred_sidechain

        return outputs

class ConfidenceModel(BaseModel):
    def __init__(self, args):
        super(ConfidenceModel, self).__init__(args, confidence_mode=True, num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1)

    def forward(self, batch):
        # move graphs to cuda
        logits = self.encoder(batch)

        return logits
    

