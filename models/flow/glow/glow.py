import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flow.glow.act_norm import ActNorm
from models.flow.glow.coupling import Coupling
from models.flow.glow.inv_conv import Invertible1x1Conv2d


class Glow(nn.Module):
    """Glow Model

    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Args:
        num_channels (int): Number of channels in middle convolution of each
            step of flow.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
        no_latent (bool): Z is not a latent variable with a prior, but instead
            it is another domain. E.g., in CycleFlow.
        un_normalize_x (bool): Un-normalize inputs before converting to logits.
    """
    def __init__(self, num_channels, num_levels, num_steps, no_latent=False, un_normalize_x=True):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)
        self.no_latent = no_latent
        self.un_normalize_x = un_normalize_x

    def forward(self, x, reverse=False, is_latent_input=False):
        if (self.no_latent or not reverse) and not is_latent_input:
            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)
        else:
            sldj = torch.zeros(x.size(0), device=x.device)

        x = squeeze(x)
        x, sldj = self.flows(x, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj, None, None

    def _pre_process(self, x):
        """De-quantize and convert the input image `x` to logits.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Logits of `x`.
            ldj (torch.Tensor): Log-determinant of the Jacobian of the transform.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        if self.un_normalize_x:
            x = x * 0.5 + 0.5

        # Expect inputs in [0, 1]
        if x.min() < 0 or x.max() > 1:
            raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                             .format(x.min(), x.max()))

        # De-quantize
        x = (x * 255. + torch.rand_like(x)) / 256.

        # Convert to logits
        y = (2 * x - 1) * self.bounds  # [-0.9, 0.9]
        y = (y + 1) / 2  # [0.05, 0.95]
        y = y.log() - (1. - y).log()  # logit

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        ldj = ldj.view(ldj.size(0), -1).sum(-1)

        return y, ldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
        idx (int): Index of the level to construct. Counter for the recursion.
    """
    def __init__(self, in_channels, mid_channels, num_levels, num_steps, idx=0):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if idx < num_levels - 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels,
                              num_steps=num_steps,
                              idx=idx + 1)
        else:
            self.next = None

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.perm = Invertible1x1Conv2d(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.perm(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.perm(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x
