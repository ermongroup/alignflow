import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum
from models.real_nvp.st_resnet import STResNet
from util import checkerboard_like


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHECKERBOARD:
            norm_channels = in_channels
            out_channels = 2 * in_channels
            in_channels = 2 * in_channels + 1
        else:
            norm_channels = in_channels // 2
            out_channels = in_channels
            in_channels = in_channels
        self.st_norm = nn.BatchNorm2d(norm_channels, affine=False)
        self.st_net = STResNet(in_channels, mid_channels, out_channels,
                               num_blocks=num_blocks, kernel_size=3, padding=1)

        # Learnable scale and shift for s
        self.s_scale = nn.Parameter(torch.ones(1))
        self.s_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x, sldj=None, reverse=True):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_like(x, reverse=self.reverse_mask)
            x_b = x * b
            x_b = 2. * self.st_norm(x_b)
            b = b.expand(x.size(0), -1, -1, -1)
            x_b = F.relu(torch.cat((x_b, -x_b, b), dim=1))
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift
            s = s * (1. - b)
            t = t * (1. - b)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_norm(x_id)
            st = F.relu(torch.cat((st, -st), dim=1))
            st = self.st_net(st)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj
