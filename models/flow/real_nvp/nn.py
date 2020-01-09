import torch
import torch.nn as nn
import torch.nn.functional as F

from util import concat_elu, WNConv2d


class NN(nn.Module):
    """Neural network used to parametrize s and t transformations for RealNVP.
    An `NN` is a stack of blocks, where each block consists of the following
    conv layers connected in a residual fashion:
      Conv: input -> nonlinearit -> conv3x3 -> nonlinearity -> gate
    where gate refers to a 1×1 convolution that doubles the number of channels,
    followed by a gated linear unit (Dauphin et al., 2016).
    The convolutional layer is identical to the one used by PixelCNN++
    (Salimans et al., 2017).

    Args:
        in_channels (int): Number of channels in the input.
        num_channels (int): Number of channels in each block of the network.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of blocks in the network.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, drop_prob):
        super(NN, self).__init__()
        conv_blocks = [WNConv2d(in_channels, mid_channels, kernel_size=3, padding=1)]
        conv_blocks += [ConvBlock(mid_channels, drop_prob)
                        for _ in range(num_blocks)]
        conv_blocks += [WNConv2d(mid_channels, out_channels, kernel_size=3, padding=1)]

        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x, aux=None):
        x = self.conv_blocks(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, num_channels, drop_prob):
        super(ConvBlock, self).__init__()
        self.conv = GatedConv(num_channels, num_channels, drop_prob)
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x


class GatedConv(nn.Module):
    """Gated Convolution Block
    Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, in_channels, out_channels, drop_prob=0.):
        super(GatedConv, self).__init__()
        self.nlin = concat_elu
        self.conv = WNConv2d(2 * in_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(drop_prob)
        self.gate = Gate(2 * out_channels, 2 * out_channels)

    def forward(self, x):
        x = self.nlin(x)
        x = self.conv(x)
        x = self.nlin(x)
        x = self.drop(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)

        return x


class Gate(nn.Module):
    def __init__(self, in_channels, out_channels, init_zeros=False):
        super(Gate, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if init_zeros:
            # Initialize second half of channels to output -10
            nn.init.constant_(self.conv.weight[out_channels // 2:], 0)
            nn.init.constant_(self.conv.bias[out_channels // 2:], -10)

    def forward(self, x):
        x = self.conv(x)

        return x


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_pad = nn.ReflectionPad2d(1)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=True)

    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = self.in_pad(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = self.out_pad(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip

        return x


class ResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
    """
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, kernel_size, padding):
        super(ResNet, self).__init__()
        self.in_pad = nn.ReflectionPad2d(padding)
        self.in_conv = WNConv2d(in_channels, mid_channels, kernel_size, padding=0, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_pad(x)
        x = self.in_conv(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x
