import torch
import torch.nn as nn

from util import mean_dim


class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).
    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width, trainable):
        super(_BaseNorm, self).__init__()

        # Input gets concatenated along channel axis
        num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.trainable = trainable
        if self.trainable:
            self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
            self.log_std = nn.Parameter(torch.ones(1, num_channels, height, width))
        else:
            self.is_initialized += 1.
        self.eps = 1e-5
        self.log_scale = 1.

    def initialize_parameters(self, x, reverse):
        if not self.training:
            return

        with torch.no_grad():
            mean, log_std = self._get_moments(x)
            self.mean.data.copy_(mean.data)
            self.log_std.data.copy_(log_std.data)
            if not reverse:
                # Only count the initialization in forward direction
                self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, sldj=None, reverse=False):
        x = torch.cat(x, dim=1)
        if not self.is_initialized:
            self.initialize_parameters(x, reverse)

        if reverse:
            x, sldj = self._scale(x, sldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, sldj = self._scale(x, sldj, reverse)
        x = x.chunk(2, dim=1)

        return x, sldj


class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow
    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and log_std are trainable parameters.
    """
    def __init__(self, num_channels, trainable=True):
        super(ActNorm, self).__init__(num_channels, 1, 1, trainable)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        log_std = torch.log(1. / (var.sqrt() + self.eps) / self.log_scale) * self.log_scale

        return mean, log_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x * self.log_std.mul(-1).exp()
            sldj = sldj - self.log_std.sum() * x.size(2) * x.size(3)
        else:
            x = x * self.log_std.exp()
            sldj = sldj + self.log_std.sum() * x.size(2) * x.size(3)

        return x, sldj


class PixNorm(_BaseNorm):
    """Pixel-wise Activation Normalization used in Flow++
    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    init, mean and log_std are trainable parameters.
    """
    def _get_moments(self, x):
        mean = torch.mean(x.clone(), dim=0, keepdim=True)
        var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
        log_std = torch.log(1. / (var.sqrt() + self.eps) / self.log_scale) * self.log_scale

        return mean, log_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x * self.log_std.mul(-1).exp()
            sldj = sldj - self.log_std.sum()
        else:
            x = x * self.inv_std.exp()
            sldj = sldj + self.log_std.sum()

        return x, sldj
