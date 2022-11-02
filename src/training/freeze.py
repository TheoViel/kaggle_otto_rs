import torch


def freeze(model, prefix=""):
    """
    Freezes a model

    Arguments:
        model {torch model} -- Model to freeze
    """
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = False

    for name, module in model.named_modules():
        if name.startswith(prefix):
            module.eval()


def unfreeze(model, prefix=""):
    """
    Unfreezes a model

    Arguments:
        modem {torch model} -- Model to unfreeze
    """
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = True

    for name, module in model.named_modules():
        if name.startswith(prefix):
            module.train()


def freeze_batchnorm(model):
    """
    Freezes the batch normalization layers of a model.
    Args:
        model (torch model): Model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
