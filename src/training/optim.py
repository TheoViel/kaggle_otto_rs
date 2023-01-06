import math
import torch
import warnings
from torch import Tensor
from itertools import chain
from torch.optim import Optimizer
from collections import defaultdict
from typing import Iterable, Union, Callable, Optional, Tuple


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


def define_optimizer(name, params, lr=1e-3, betas=(0.9, 0.999)):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.
    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.
    Raises:
        NotImplementedError: Specified optimizer name is not supported.
    Returns:
        torch optimizer: Optimizer
    """

    if name.lower() == "ranger":
        radam = getattr(torch.optim, "RAdam")(params, lr=lr, betas=betas)
        return Lookahead(radam, alpha=0.5, k=5)
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr, betas=betas)
    except AttributeError:
        raise NotImplementedError(name)

    return optimizer
