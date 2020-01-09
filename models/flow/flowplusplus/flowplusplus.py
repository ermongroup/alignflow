import math
import torch
import torch.nn as nn

from models.flow.util import (channelwise, checkerboard, safe_log,
                              squeeze, tanh_to_logits, unsqueeze)
from models.flow.flowplusplus.act_norm import ActNorm
from models.flow.flowplusplus.flip import Flip
from models.flow.flowplusplus.inv_conv import InvConv
from models.flow.flowplusplus.nn import GatedConv
from models.flow.flowplusplus.coupling import Coupling


class FlowPlusPlus(nn.Module):
    """Flow++ Model modified for CycleFlow

    Based on the paper:
    "Flow++: Improving Flow-Based Generative Models
        with Variational Dequantization and Architecture Design"
    by Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel
    (https://openreview.net/forum?id=Hyg74h05tX).

    Args:
        scales (tuple or list): Number of each type of coupling layer in each
            scale. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_shape (tuple): Shape of a single input in the batch.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_dequant_blocks (int): Number of blocks in the dequantization flows.
        use_attn (bool): Use attention in the coupling layers.
        symmetric (bool): Use a symmetric model (two Flow++'s head-to-head).
        is_img2img (bool): Model used for image-to-image translation.
    """
    def __init__(self,
                 scales=((0, 4), (2, 3)),
                 in_shape=(3, 32, 32),
                 mid_channels=96,
                 num_blocks=10,
                 num_dequant_blocks=2,
                 num_components=32,
                 drop_prob=0.2,
                 use_attn=True,
                 symmetric=False,
                 is_img2img=True):
        super(FlowPlusPlus, self).__init__()
        # Register bounds to pre-process images, not learnable
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        if num_dequant_blocks > 0:
            self.dequant_fwd = _Dequantization(in_shape=in_shape,
                                               mid_channels=mid_channels,
                                               num_blocks=num_dequant_blocks,
                                               num_components=num_components,
                                               use_attn=use_attn,
                                               drop_prob=drop_prob,
                                               num_flows=3,
                                               aux_channels=16)
            self.dequant_rev = _Dequantization(in_shape=in_shape,
                                               mid_channels=mid_channels,
                                               num_blocks=num_dequant_blocks,
                                               use_attn=use_attn,
                                               drop_prob=drop_prob,
                                               num_flows=3,
                                               aux_channels=16)
        else:
            self.dequant_fwd = self.dequant_rev = None

        self.is_img2img = is_img2img
        self.symmetric = symmetric
        if self.symmetric:
            self.flows = _SymmetricFlows(scales=scales,
                                         in_shape=in_shape,
                                         mid_channels=mid_channels,
                                         num_blocks=num_blocks,
                                         num_components=num_components,
                                         use_attn=use_attn,
                                         drop_prob=drop_prob)
        else:
            self.flows = _Flows(scales=scales,
                                in_shape=in_shape,
                                mid_channels=mid_channels,
                                num_blocks=num_blocks,
                                num_components=num_components,
                                use_attn=use_attn,
                                drop_prob=drop_prob)

    def forward(self, x, reverse=False, is_latent_input=False):
        # Whether input/output domains are image domains
        is_image_input = not is_latent_input and (self.is_img2img or not reverse)
        is_image_output = self.is_img2img or reverse

        # Dequantize and convert to logits
        sldj_x = torch.zeros(x.size(0), device=x.device)
        if is_image_input:
            x, sldj_x = self.dequantize(x, sldj_x, reverse)
            x, sldj_x = tanh_to_logits(x, sldj_x, reverse=False)

        # Apply flows
        if self.symmetric:
            x, sldj_x, z, sldj_z = self.flows(x, sldj_x, reverse, is_latent_input)
        else:
            x, sldj_x = self.flows(x, sldj_x, reverse, is_latent_input)
            z, sldj_z = None, None  # No shared latent space

        # Convert to image
        if is_image_output:
            x, sldj_x = tanh_to_logits(x, sldj_x, reverse=True)

        return x, sldj_x, z, sldj_z

    def dequantize(self, x, sldj, reverse=False):
        if not reverse and self.dequant_fwd is not None:
            x, sldj = self.dequant_fwd(x, sldj)
        elif reverse and self.dequant_rev is not None:
            x, sldj = self.dequant_rev(x, sldj)
        elif self.symmetric or not self.is_img2img:
            x = (x * 255. + torch.rand_like(x)) / 256.
        else:
            pass  # No dequantization for non-symmetric model (no MLE training)

        return x, sldj


class _SymmetricFlows(nn.Module):
    """Recursive builder for a symmetric Flow++ model.

    Each constructed `_SymmetricFlows` corresponds to a single scale in Flow++.
    The constructor is recursively called to build a full model.

    Args:
        scales (tuple): Number of each type of coupling layer in each scale.
            Each scale is a 2-tuple of the form (num_channelwise, num_checkerboard).
        in_shape (tuple): Shape of a single input in the batch.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_components (int): Number of components in the mixture.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, scales, in_shape, mid_channels, num_blocks, num_components, use_attn, drop_prob):
        super(_SymmetricFlows, self).__init__()
        self.x2z_flow = _Flows(scales=scales,
                               in_shape=in_shape,
                               mid_channels=mid_channels,
                               num_blocks=num_blocks,
                               num_components=num_components,
                               use_attn=use_attn,
                               drop_prob=drop_prob)
        self.y2z_flow = _Flows(scales=scales,
                               in_shape=in_shape,
                               mid_channels=mid_channels,
                               num_blocks=num_blocks,
                               num_components=num_components,
                               use_attn=use_attn,
                               drop_prob=drop_prob)

    def forward(self, x, sldj_x, reverse=False, is_latent_input=False):
        if reverse:
            if is_latent_input:
                z, sldj_z = x, sldj_x
            else:
                z, sldj_z = self.y2z_flow(x, sldj_x, reverse=False)
            x, sldj_x = self.x2z_flow(z, sldj_z, reverse=True)
        else:
            if is_latent_input:
                z, sldj_z = x, sldj_x
            else:
                z, sldj_z = self.x2z_flow(x, sldj_x, reverse=False)
            x, sldj_x = self.y2z_flow(z, sldj_z, reverse=True)

        return x, sldj_x, z, sldj_z


class _Flows(nn.Module):
    """Recursive builder for a Flow++ model.

    Each constructed `_Flows` corresponds to a single scale in Flow++.
    The constructor is recursively called to build a full model.

    Args:
        scales (tuple): Number of each type of coupling layer in each scale.
            Each scale is a 2-tuple of the form (num_channelwise, num_checkerboard).
        in_shape (tuple): Shape of a single input in the batch.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_components (int): Number of components in the mixture.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, scales, in_shape, mid_channels, num_blocks, num_components, use_attn, drop_prob):
        super(_Flows, self).__init__()
        in_channels, in_height, in_width = in_shape
        num_channelwise, num_checkerboard = scales[0]
        channels = []
        for i in range(num_channelwise):
            channels += [ActNorm(in_channels // 2),
                         InvConv(in_channels // 2),
                         Coupling(in_channels=in_channels // 2,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob),
                         Flip()]

        checkers = []
        for i in range(num_checkerboard):
            checkers += [ActNorm(in_channels),
                         InvConv(in_channels),
                         Coupling(in_channels=in_channels,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob),
                         Flip()]
        self.channels = nn.ModuleList(channels) if channels else None
        self.checkers = nn.ModuleList(checkers) if checkers else None

        if len(scales) <= 1:
            self.next = None
        else:
            next_shape = (2 * in_channels, in_height // 2, in_width // 2)
            self.next = _Flows(scales=scales[1:],
                               in_shape=next_shape,
                               mid_channels=mid_channels,
                               num_blocks=num_blocks,
                               num_components=num_components,
                               use_attn=use_attn,
                               drop_prob=drop_prob)

    def forward(self, x, sldj, reverse=False, is_latent_input=False):
        if is_latent_input:
            raise NotImplementedError('_Flows should not handle is_latent_input')

        if reverse:
            if self.next is not None:
                x = squeeze(x)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)

            if self.checkers:
                x = checkerboard(x)
                for flow in reversed(self.checkers):
                    x, sldj = flow(x, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.channels:
                x = channelwise(x)
                for flow in reversed(self.channels):
                    x, sldj = flow(x, sldj, reverse)
                x = channelwise(x, reverse=True)
        else:
            if self.channels:
                x = channelwise(x)
                for flow in self.channels:
                    x, sldj = flow(x, sldj, reverse)
                x = channelwise(x, reverse=True)

            if self.checkers:
                x = checkerboard(x)
                for flow in self.checkers:
                    x, sldj = flow(x, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.next:
                x = squeeze(x)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)

        return x, sldj


class _Dequantization(nn.Module):
    """Dequantization Network for Flow++

    Args:
        in_shape (tuple): Shape of the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        drop_prob (float): Dropout probability.
        num_flows (int): Number of InvConv+MLCoupling flows to use.
        aux_channels (int): Number of channels in auxiliary input to couplings.
        num_components (int): Number of components in the mixture.
    """
    def __init__(self, in_shape, mid_channels, num_blocks, drop_prob,
                 use_attn=True, num_flows=4, aux_channels=32, num_components=32):
        super(_Dequantization, self).__init__()
        in_channels, in_height, in_width = in_shape
        self.aux_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, aux_channels, kernel_size=3, padding=1),
            GatedConv(aux_channels, drop_prob),
            GatedConv(aux_channels, drop_prob),
            GatedConv(aux_channels, drop_prob))

        flows = []
        for _ in range(num_flows):
            flows += [ActNorm(in_channels),
                      InvConv(in_channels),
                      Coupling(in_channels, mid_channels, num_blocks,
                               num_components, drop_prob,
                               use_attn=use_attn,
                               aux_channels=aux_channels),
                      Flip()]
        self.flows = nn.ModuleList(flows)

    def forward(self, x, sldj):
        u = torch.randn_like(x)
        eps_nll = 0.5 * (u ** 2 + math.log(2 * math.pi))

        aux = self.aux_conv(torch.cat(checkerboard(x), dim=1))
        u = checkerboard(u)
        for i, flow in enumerate(self.flows):
            u, sldj = flow(u, sldj, aux=aux) if i % 4 == 2 else flow(u, sldj)
        u = checkerboard(u, reverse=True)

        u = torch.sigmoid(u)
        x = (x * 255. + u) / 256.

        sigmoid_ldj = safe_log(u) + safe_log(1. - u)
        sldj = sldj + (eps_nll + sigmoid_ldj).flatten(1).sum(-1)

        return x, sldj
