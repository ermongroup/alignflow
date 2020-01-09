import torch
import torch.nn as nn


class Shuffle(nn.Module):
    """Shuffle channels in a fixed random order. This idea was introduced
    in the Glow paper as an intermediate design between Real NVP's bipartite
    graph of alternating channel dependencies, and Glow's 1x1 invertible
    convolutions.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Shuffle, self).__init__()
        # Register buffers to include the permutation in checkpoints
        num_channels *= 2  # Treat the 2-tuple as a concatenated tensor
        self.register_buffer('fwd_idxs', torch.randperm(num_channels))
        self.register_buffer('rev_idxs', torch.zeros_like(self.fwd_idxs))
        for i in range(num_channels):
            self.rev_idxs[self.fwd_idxs[i]] = i

    def forward(self, x, sldj_x, reverse=False):
        x = torch.cat(x, dim=1)
        assert x.size(1) == len(self.fwd_idxs), 'Mismatched number of channels'

        if reverse:
            x = x[:, self.rev_idxs, ...]
        else:
            x = x[:, self.fwd_idxs, ...]

        return x.chunk(2, dim=1), sldj_x
