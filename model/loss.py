import torch.nn.functional as F


def binary_cross_entropy(output, target):
    return F.binary_cross_entropy(output, target)


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_logit_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
