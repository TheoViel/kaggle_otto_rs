import os
import re
import torch
import random
import warnings
import numpy as np


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".
        strict (bool, optional): Whether to allow missing/additional keys. Defaults to False.

    Returns:
        torch model: Model with loaded weights.
    """
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")

    try:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=strict,
        )
    except RuntimeError:
        model.encoder.fc = torch.nn.Linear(model.nb_ft, 1)
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=strict,
        )

    return model


def load_pretrained_weights(model, filename, verbose=1, cp_folder=""):
    """
    Loads the weights of a pretrained transformer.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".

    Returns:
        torch model: Model with loaded weights.
    """
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder, filename)}\n")

    weights = torch.load(os.path.join(cp_folder, filename))

    # Rename if MLM pretraining
    weights = {re.sub(f"{k.split('.')[0]}.", 'transformer.', k): v for k, v in weights.items()}

    errors = model.load_state_dict(weights, strict=False)

    if len(errors.unexpected_keys) > 10:
        warnings.warn(f"Too many unexpected_keys keys : \n {errors.unexpected_keys}", UserWarning)


def count_parameters(model, all=False):
    """
    Counts the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_grad_norm(model):
    """
    TODO

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    return torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]
            ),
        2
    )
