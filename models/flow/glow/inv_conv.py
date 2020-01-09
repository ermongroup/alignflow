import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Invertible1x1Conv2d(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, num_channels):
        super(Invertible1x1Conv2d, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.filters = nn.Parameter(torch.from_numpy(w_init))

    def get_weight(self, x, reverse):
        ldj = torch.slogdet(self.filters)[1] * x.size(2) * x.size(3)
        if reverse:
            weight = torch.inverse(self.filters.double()).float()
        else:
            weight = self.filters
        weight = weight.view(self.num_channels, self.num_channels, 1, 1)

        return weight, ldj

    def forward(self, x, sldj, reverse=False):
        weight, ldj = self.get_weight(x, reverse)
        z = F.conv2d(x, weight)
        if reverse:
            sldj = sldj - ldj
        else:
            sldj = sldj + ldj

        return z, sldj
