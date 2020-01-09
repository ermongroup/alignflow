import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flow.real_nvp.nn import ResNet, NN
from util import BatchNormSLDJ


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.
    Args:
        in_channels (int): Number of channels in each of x_change, x_id.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        use_fpp_nn (bool): Use Flow++-style NN.
        use_out_norm (bool): Use BatchNorm on the coupling layer output.
    """
    def __init__(self, in_channels, mid_channels, num_blocks,
                 use_fpp_nn=True, use_out_norm=False):
        super(CouplingLayer, self).__init__()
        if use_fpp_nn:
            self.st_norm = None
            self.st_net = NN(in_channels=in_channels,
                             mid_channels=mid_channels,
                             out_channels=2 * in_channels,
                             num_blocks=num_blocks,
                             drop_prob=0.)
        else:
            self.st_norm = nn.BatchNorm2d(in_channels, affine=False)
            self.st_net = ResNet(in_channels=2 * in_channels,
                                 mid_channels=mid_channels,
                                 out_channels=2 * in_channels,
                                 num_blocks=num_blocks,
                                 kernel_size=3,
                                 padding=1)
        if use_out_norm:
            self.out_norm = BatchNormSLDJ(in_channels)
        else:
            self.out_norm = None

        # Learnable scale and shift for s
        self.s_scale = nn.Parameter(torch.ones(1))
        self.s_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x, sldj=None, reverse=True):
        x_change, x_id = x
        if self.st_norm:
            st_in = self.st_norm(x_id)
            st = self.st_net(F.relu(torch.cat((st_in, -st_in), dim=1)))
        else:
            st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = self.s_scale * torch.tanh(s) + self.s_shift

        # Scale and translate
        if reverse:
            if self.out_norm:
                x_change, sldj = self.out_norm(x_change, sldj, reverse, training=False)

            x_change = (x_change - t) * s.mul(-1).exp()
            sldj = sldj - s.flatten(1).sum(1)
        else:
            x_change = (x_change * s.exp() + t)
            sldj = sldj + s.flatten(1).sum(1)

            if self.out_norm:
                x_change, sldj = self.out_norm(x_change, sldj, reverse, training=self.training)

        x = (x_change, x_id)

        return x, sldj
