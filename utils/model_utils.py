import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(param.numel() for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = True
    return module
