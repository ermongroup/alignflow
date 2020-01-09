import torch.nn.functional as F


def mse(inputs, targets):
    """Mean-squared error between `inputs` and `targets`.

    Args:
        inputs (torch.Tensor): Images predicted by the model.
        targets (torch.Tensor): Target images.

    Returns:
        Mean-squared error in (-1, 1) space, element-wise mean.
    """
    return F.mse_loss(inputs, targets)
