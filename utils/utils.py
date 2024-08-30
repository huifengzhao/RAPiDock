##########################################################################
# File Name: utils.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Tue 24 Oct 2023 01:16:36 PM CST
#########################################################################

import os
import yaml
import torch
from torch_geometric.nn.data_parallel import DataParallel
from models.model import ConfidenceModel, ScoreModel


def save_yaml_file(path, content):
    assert isinstance(
        path, str
    ), f"path must be a string, got {path} which is a {type(path)}"
    content = yaml.dump(data=content)
    if (
        "/" in path
        and os.path.dirname(path)
        and not os.path.exists(os.path.dirname(path))
    ):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)


def get_model(args, confidence_mode=False, no_parallel=False):
    if confidence_mode:
        model_class = ConfidenceModel
    else:
        model_class = ScoreModel
    model = model_class(args)

    model = model.cuda(args.gpu)
    if torch.cuda.is_available() and not no_parallel:
        device_ids = [args.gpu + i for i in range(args.num_gpu)]
        model = DataParallel(model, device_ids=device_ids)
    return model


def get_optimizer_and_scheduler(args, model, scheduler_mode="min"):
    """
    Initialize optimizer and load if applicable
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.w_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=0.7,
            patience=args.scheduler_patience,
            min_lr=args.lr / 100,
        )
    else:
        print("No scheduler")
        scheduler = None

    return optimizer, scheduler


class ExponentialMovingAverage:
    """from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters."""

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [
            tensor.to(device) for tensor in state_dict["shadow_params"]
        ]
