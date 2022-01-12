import torch

def Delta(pred, target, exp=1.0):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** exp).float().mean()

def Delta1(pred, target):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** 1).float().mean()