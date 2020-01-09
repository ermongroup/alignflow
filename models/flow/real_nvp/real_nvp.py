import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flow.real_nvp.act_norm import ActNorm
from models.flow.real_nvp.coupling_layer import CouplingLayer
from models.flow.real_nvp.inv_conv import InvConv
from models.flow.real_nvp.shuffle import Shuffle
from models.flow.util import (channelwise, checkerboard,
                              alt_squeeze, squeeze, unsqueeze)

MAX_CHANNELS = 256  # Upper bound on mid_channels in coupling layers


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        scales (tuple or list): Number of each type of coupling layer in each
            scale. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        norm_type (str): Type of normalization to use. One of 'group', 'instance', 'batch'.
        symmetric (bool): Use symmetric architecture instead of original Real NVP.
        is_img2img (bool): Model used for image-to-image translation.
        mid_scales (tuple or list, optional): Scale config for 3-head symmetric
            RealNVP. See `scales` for format.
        use_split (bool): Split out half the features after each squeeze.
        use_fpp_nn (bool): Use Flow++-style NN in the coupling layers.
    """
    def __init__(self, scales=((3, 4), (0, 4)),
                 in_channels=3, mid_channels=64, num_blocks=8,
                 norm_type='batch', symmetric=False, is_img2img=True,
                 mid_scales=None, use_split=True, use_fpp_nn=False):
        super(RealNVP, self).__init__()
        # Register bounds to pre-process images, not learnable
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        self.is_img2img = is_img2img

        # Get inner layers
        self.symmetric = symmetric
        if self.symmetric:
            self.flows = _SymmetricRealNVP(in_scales=scales,
                                           mid_scales=mid_scales,
                                           in_channels=in_channels,
                                           mid_channels=mid_channels,
                                           num_blocks=num_blocks,
                                           norm_type=norm_type,
                                           use_split=use_split,
                                           use_fpp_nn=use_fpp_nn)
        else:
            self.flows = _RealNVP(scales=scales,
                                  in_channels=in_channels,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  norm_type=norm_type,
                                  use_split=use_split,
                                  use_fpp_nn=use_fpp_nn)

    def forward(self, x, reverse=False, is_latent_input=False):
        # Whether input/output domains are image domains
        is_image_input = not is_latent_input and (self.is_img2img or not reverse)
        is_image_output = self.is_img2img or reverse

        # Dequantize and convert to logits
        if is_image_input:
            x, sldj_x = self._pre_process(x)
        else:
            sldj_x = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)

        # Apply flows
        if self.symmetric:
            x, sldj_x, z, sldj_z = self.flows(x, sldj_x, reverse, is_latent_input)
        else:
            x, sldj_x = self.flows(x, sldj_x, reverse, is_latent_input)
            z, sldj_z = None, None  # No shared latent space

        # Convert to image
        if is_image_output:
            x = torch.tanh(x)
            sldj_x = sldj_x + torch.log(1. - x ** 2).flatten(1).sum(1)

        return x, sldj_x, z, sldj_z

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.
        Args:
            x (torch.Tensor): Input image.
        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = x * 0.5 + 0.5
        y = (y * 255. + torch.rand_like(y)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(1)

        return y, sldj


class _RealNVP(nn.Module):
    """Recursive constructor for a `RealNVP` model.

    Each `_RealNVP` corresponds to a single scale coupling layers
    in `RealNVP`. The constructor is recursively called to build a full
    `RealNVP` model. The recursive structure makes it easy to split/squeeze,
    pass to next block, then unsqueeze/concat.

    Args:
        scales (tuple or list): Number of each type of coupling layer in each
            scale. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        norm_type (str): Type of normalization to use. One of 'group' or 'batch'.
        use_act_norm (bool): Use activation normalization after each coupling layer.
        use_inv_conv (bool): Use invertible convolution after each coupling layer.
        use_shuffle (bool): Use channel-shuffle rather than flip.
        skip_final_flip (bool): Skip the last flip in the network. Used for
            symmetric RealNVP.


    Returns:
        x (torch.Tensor): Output tensor.
        sldj (torch.Tensor): Sum of log-determinants of Jacobians of all transformations.
    """
    def __init__(self, scales, in_channels, mid_channels, num_blocks, norm_type,
                 use_act_norm=True,
                 use_inv_conv=False,
                 use_shuffle=False,
                 use_split=True,
                 use_fpp_nn=False,
                 skip_final_flip=False):
        super(_RealNVP, self).__init__()
        self.use_split = use_split

        # Get scale configuration
        num_channelwise, num_checkerboard = scales[0]

        # Checkerboard-mask coupling layers
        checkers = []
        for _ in range(num_checkerboard):
            if use_act_norm:
                checkers.append(ActNorm(in_channels))
            if use_inv_conv:
                checkers.append(InvConv(in_channels))
            checkers.append(CouplingLayer(in_channels, mid_channels, num_blocks, use_fpp_nn))
            if use_shuffle:
                checkers.append(Shuffle(in_channels))
            else:
                checkers.append(Flip())
        if num_channelwise == 0 and checkers and skip_final_flip:
            _ = checkers.pop()
        self.checkers = nn.ModuleList(checkers) if num_checkerboard > 0 else None

        # Channelwise-mask coupling layers
        next_mid_channels = min(MAX_CHANNELS, mid_channels * (2 if use_split else 1))
        in_channels *= 2
        channels = []
        for _ in range(num_channelwise):
            if use_act_norm:
                channels.append(ActNorm(in_channels))
            if use_inv_conv:
                channels.append(InvConv(in_channels))
            channels.append(CouplingLayer(in_channels, next_mid_channels, num_blocks, use_fpp_nn))
            if use_shuffle:
                channels.append(Shuffle(in_channels))
            else:
                channels.append(Flip())
        if channels and skip_final_flip:
            _ = channels.pop()
        self.channels = nn.ModuleList(channels) if num_channelwise > 0 else None

        if len(scales) == 1:
            self.next = None
        else:
            self.next = _RealNVP(scales=scales[1:],
                                 in_channels=in_channels * (1 if use_split else 2),
                                 mid_channels=next_mid_channels,
                                 num_blocks=num_blocks,
                                 norm_type=norm_type,
                                 use_shuffle=use_shuffle,
                                 use_split=use_split,
                                 use_fpp_nn=use_fpp_nn,
                                 skip_final_flip=skip_final_flip)

    def forward(self, x, sldj, reverse=False, is_latent_input=False):
        if reverse:
            if self.next:
                # Re-squeeze -> split -> next block
                x = alt_squeeze(x)
                if self.use_split:
                    x, x_split = x.chunk(2, dim=1)
                    x, sldj = self.next(x, sldj, reverse)
                    x = torch.cat((x, x_split), dim=1)
                else:
                    x, sldj = self.next(x, sldj, reverse)
                x = alt_squeeze(x, reverse=True)

            if self.channels:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze(x)
                x = channelwise(x)
                for coupling in reversed(self.channels):
                    x, sldj = coupling(x, sldj, reverse)
                x = channelwise(x, reverse=True)
                x = unsqueeze(x)

            if self.checkers:
                x = checkerboard(x)
                for coupling in reversed(self.checkers):
                    x, sldj = coupling(x, sldj, reverse)
                x = checkerboard(x, reverse=True)
        else:
            if self.checkers:
                x = checkerboard(x)
                for coupling in self.checkers:
                    x, sldj = coupling(x, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.channels:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze(x)
                x = channelwise(x)
                for coupling in self.channels:
                    x, sldj = coupling(x, sldj, reverse)
                x = channelwise(x, reverse=True)
                x = unsqueeze(x)

            if self.next:
                # Re-squeeze -> split -> next block
                x = alt_squeeze(x)
                if self.use_split:
                    x, x_split = x.chunk(2, dim=1)
                    x, sldj = self.next(x, sldj, reverse)
                    x = torch.cat((x, x_split), dim=1)
                else:
                    x, sldj = self.next(x, sldj, reverse)
                x = alt_squeeze(x, reverse=True)

        return x, sldj


class _SymmetricRealNVP(nn.Module):
    """Constructor for symmetric `RealNVP` model.

    Args:
        in_scales (tuple or list): Scale configurations for the in_flow and
            out_flow models. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_scales (tuple or list): Same as in_scales, but used for mid_flow.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        norm_type (str): Type of normalization to use. One of 'group' or 'batch'.

    Returns:
        scales (list): List of tuples, (x, sldj) for each scale in the Real NVP model.
    """
    def __init__(self, in_scales, mid_scales, in_channels, mid_channels,
                 num_blocks, norm_type, use_split, use_fpp_nn):
        super(_SymmetricRealNVP, self).__init__()
        # Diagram of Symmetric RealNVP (arrows are flow models)
        #       Z
        #       ^
        #       ^
        # X <-> H <-> Y
        self.in_flow = _RealNVP(scales=in_scales,
                                in_channels=in_channels,
                                mid_channels=mid_channels,
                                num_blocks=num_blocks,
                                norm_type=norm_type,
                                use_split=use_split,
                                use_fpp_nn=use_fpp_nn)

        if mid_scales:
            self.mid_flow = _RealNVP(scales=mid_scales,
                                     in_channels=in_channels,
                                     mid_channels=mid_channels,
                                     num_blocks=num_blocks,
                                     norm_type=norm_type,
                                     use_split=use_split,
                                     use_fpp_nn=use_fpp_nn)
        else:
            self.mid_flow = None

        self.out_flow = _RealNVP(scales=in_scales,
                                 in_channels=in_channels,
                                 mid_channels=mid_channels,
                                 num_blocks=num_blocks,
                                 norm_type=norm_type,
                                 use_split=use_split,
                                 use_fpp_nn=use_fpp_nn,
                                 skip_final_flip=True)

    def forward(self, x, sldj_x, reverse=False, is_latent_input=False):
        if reverse:
            if is_latent_input:
                z, sldj_z = x, sldj_x
                h, sldj_h = self.mid_flow(z, sldj_z, reverse=True) if self.mid_flow else (z, sldj_z)
            else:
                h, sldj_h = self.out_flow(x, sldj_x, reverse=False)
                z, sldj_z = self.mid_flow(h, sldj_h, reverse=False) if self.mid_flow else (h, sldj_h)

            x, sldj_x = self.in_flow(h, sldj_h, reverse=True)
        else:
            if is_latent_input:
                z, sldj_z = x, sldj_x
                h, sldj_h = self.mid_flow(z, sldj_z, reverse=True) if self.mid_flow else (z, sldj_z)
            else:
                h, sldj_h = self.in_flow(x, sldj_x, reverse=False)
                z, sldj_z = self.mid_flow(h, sldj_h, reverse=False) if self.mid_flow else (h, sldj_h)
            x, sldj_x = self.out_flow(h, sldj_h, reverse=True)

        return x, sldj_x, z, sldj_z


class Flip(nn.Module):
    def forward(self, x, sldj, reverse=False):
        assert isinstance(x, tuple) and len(x) == 2
        return (x[1], x[0]), sldj
