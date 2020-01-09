import torch.nn as nn


class Flip(nn.Module):
    def forward(self, x, sldj, reverse=False):
        assert isinstance(x, tuple) and len(x) == 2
        return (x[1], x[0]), sldj
