import torch
import torch.nn.functional as F


def rmse_loss(predict, target):

    return torch.sqrt(F.mse_loss(predict, target))