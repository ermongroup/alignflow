import os
import torch
import torch.nn as nn
import util

from itertools import chain
from models.patch_gan import PatchGAN
from models.real_nvp import RealNVP, RealNVPLoss


class Flow2Flow(nn.Module):
    """Flow2Flow Model

    Normalizing flows for unpaired image-to-image translation.
    Uses two normalizing flow models (RealNVP) for the generators,
    and two PatchGAN discriminators. The generators map to a shared
    intermediate latent space `Z` with simple prior `p_Z`, and the
    whole system optimizes a hybrid GAN-MLE objective.
    """
    def __init__(self, args):
        """
        Args:
            args: Configuration args passed in via the command line.
        """
        super(Flow2Flow, self).__init__()
        self.device = 'cuda' if len(args.gpu_ids) > 0 else 'cpu'
        self.gpu_ids = args.gpu_ids
        self.is_training = args.is_training

        self.in_channels = args.num_channels
        self.out_channels = 4 ** (args.num_scales - 1) * self.in_channels

        # Set up RealNVP generators (g_src: X <-> Z, g_tgt: Y <-> Z)
        self.g_src = RealNVP(num_scales=args.num_scales,
                             in_channels=args.num_channels,
                             mid_channels=args.num_channels_g,
                             num_blocks=args.num_blocks,
                             un_normalize_x=True,
                             no_latent=False)
        util.init_model(self.g_src, init_method=args.initializer)
        self.g_tgt = RealNVP(num_scales=args.num_scales,
                             in_channels=args.num_channels,
                             mid_channels=args.num_channels_g,
                             num_blocks=args.num_blocks,
                             un_normalize_x=True,
                             no_latent=False)
        util.init_model(self.g_tgt, init_method=args.initializer)

        if self.is_training:
            # Set up discriminators
            self.d_tgt = PatchGAN(args)  # Answers Q "is this tgt image real?"
            self.d_src = PatchGAN(args)  # Answers Q "is this src image real?"

            self._data_parallel()

            # Set up loss functions
            self.max_grad_norm = args.clip_gradient
            self.lambda_mle = args.lambda_mle
            self.mle_loss_fn = RealNVPLoss()
            self.gan_loss_fn = util.GANLoss(device=self.device, use_least_squares=True)

            self.clamp_jacobian = args.clamp_jacobian
            self.jc_loss_fn = util.JacobianClampingLoss(args.jc_lambda_min, args.jc_lambda_max)

            # Set up optimizers
            g_src_params = util.get_param_groups(self.g_src, args.weight_norm_l2, norm_suffix='weight_g')
            g_tgt_params = util.get_param_groups(self.g_tgt, args.weight_norm_l2, norm_suffix='weight_g')
            self.opt_g = torch.optim.Adam(chain(g_src_params, g_tgt_params),
                                          lr=args.rnvp_lr,
                                          betas=(args.rnvp_beta_1, args.rnvp_beta_2))
            self.opt_d = torch.optim.Adam(chain(self.d_tgt.parameters(), self.d_src.parameters()),
                                          lr=args.lr,
                                          betas=(args.beta_1, args.beta_2))
            self.optimizers = [self.opt_g, self.opt_d]
            self.schedulers = [util.get_lr_scheduler(opt, args) for opt in self.optimizers]

            # Setup image mixers
            buffer_capacity = 50 if args.use_mixer else 0
            self.src2tgt_buffer = util.ImageBuffer(buffer_capacity)  # Buffer of generated tgt images
            self.tgt2src_buffer = util.ImageBuffer(buffer_capacity)  # Buffer of generated src images
        else:
            self._data_parallel()

        # Images in flow src -> lat -> tgt
        self.src = None
        self.src2lat = None
        self.src2tgt = None

        # Images in flow tgt -> lat -> src
        self.tgt = None
        self.tgt2lat = None
        self.tgt2src = None

        # Jacobian clamping tensors
        self.src_jc = None
        self.tgt_jc = None
        self.src2tgt_jc = None
        self.tgt2src_jc = None

        # Discriminator loss
        self.loss_d_tgt = None
        self.loss_d_src = None
        self.loss_d = None

        # Generator GAN loss
        self.loss_gan_src = None
        self.loss_gan_tgt = None
        self.loss_gan = None

        # Generator MLE loss
        self.loss_mle_src = None
        self.loss_mle_tgt = None
        self.loss_mle = None

        # Jacobian Clamping loss
        self.loss_jc_src = None
        self.loss_jc_tgt = None
        self.loss_jc = None

        # Generator total loss
        self.loss_g = None

    def set_inputs(self, src_input, tgt_input=None):
        """Set the inputs prior to a forward pass through the network.

        Args:
            src_input: Tensor with src input
            tgt_input: Tensor with tgt input
        """
        self.src = src_input.to(self.device)
        if tgt_input is not None:
            self.tgt = tgt_input.to(self.device)

    def forward(self):
        """No-op. We do the forward pass in `backward_g`."""
        pass

    def test(self):
        """Run a forward pass through the generator for test inference.
        Used during test inference only, as this throws away forward-pass values,
        which would be needed for backprop.

        Important: Call `set_inputs` prior to each successive call to `test`.
        """
        # Disable auto-grad because we will not backprop
        with torch.no_grad():
            src2lat, _ = self.g_src(self.src, reverse=False)
            src2lat2tgt, _ = self.g_tgt(src2lat, reverse=True)
            self.src2tgt = torch.tanh(src2lat2tgt)

            tgt2lat, _ = self.g_tgt(self.tgt, reverse=False)
            tgt2lat2src, _ = self.g_src(tgt2lat, reverse=True)
            self.tgt2src = torch.tanh(tgt2lat2src)

    def _forward_d(self, d, real_img, fake_img):
        """Forward  pass for one discriminator."""

        # Forward on real and fake images (detach fake to avoid backprop into generators)
        loss_real = self.gan_loss_fn(d(real_img), is_tgt_real=True)
        loss_fake = self.gan_loss_fn(d(fake_img.detach()), is_tgt_real=False)
        loss_d = 0.5 * (loss_real + loss_fake)

        return loss_d

    def backward_d(self):
        # Forward tgt discriminator
        src2tgt = self.src2tgt_buffer.sample(self.src2tgt)
        self.loss_d_tgt = self._forward_d(self.d_tgt, self.tgt, src2tgt)

        # Forward src discriminator
        tgt2src = self.tgt2src_buffer.sample(self.tgt2src)
        self.loss_d_src = self._forward_d(self.d_src, self.src, tgt2src)

        # Backprop
        self.loss_d = self.loss_d_tgt + self.loss_d_src
        self.loss_d.backward()

    def backward_g(self):
        if self.clamp_jacobian:
            # Double batch size with perturbed inputs for Jacobian Clamping
            self._jc_preprocess()

        # Forward src -> lat: Get MLE loss
        self.src2lat, sldj_src2lat = self.g_src(self.src, reverse=False)
        self.loss_mle_src = self.lambda_mle * self.mle_loss_fn(self.src2lat, sldj_src2lat)

        # Finish src -> lat -> tgt: Say target is real to invert loss
        src2tgt, _ = self.g_tgt(self.src2lat, reverse=True)
        self.src2tgt = torch.tanh(src2tgt)

        # Forward tgt -> lat: Get MLE loss
        self.tgt2lat, sldj_tgt2lat = self.g_tgt(self.tgt, reverse=False)
        self.loss_mle_tgt = self.lambda_mle * self.mle_loss_fn(self.tgt2lat, sldj_tgt2lat)

        # Finish tgt -> lat -> src: Say source is real to invert loss
        tgt2src, _ = self.g_src(self.tgt2lat, reverse=True)
        self.tgt2src = torch.tanh(tgt2src)

        # Jacobian Clamping loss
        if self.clamp_jacobian:
            # Split inputs and outputs from Jacobian Clamping
            self._jc_postprocess()
            self.loss_jc_src = self.jc_loss_fn(self.src2tgt, self.src2tgt_jc, self.src, self.src_jc)
            self.loss_jc_tgt = self.jc_loss_fn(self.tgt2src, self.tgt2src_jc, self.tgt, self.tgt_jc)
            self.loss_jc = self.loss_jc_src + self.loss_jc_tgt
        else:
            self.loss_jc_src = self.loss_jc_tgt = self.loss_jc = 0.

        # GAN loss
        self.loss_gan_src = self.gan_loss_fn(self.d_tgt(self.src2tgt), is_tgt_real=True)
        self.loss_gan_tgt = self.gan_loss_fn(self.d_src(self.tgt2src), is_tgt_real=True)

        # Total losses
        self.loss_gan = self.loss_gan_src + self.loss_gan_tgt
        self.loss_mle = self.loss_mle_src + self.loss_mle_tgt

        # Backprop
        self.loss_g = self.loss_gan + self.loss_mle + self.loss_jc
        self.loss_g.backward()

    def train_iter(self):
        """Run a training iteration (forward/backward) on a single batch.
        Important: Call `set_inputs` prior to each call to this function.
        """
        # Forward
        self.forward()

        # Backprop the generators
        self.opt_g.zero_grad()
        self.backward_g()
        util.clip_grad_norm(self.opt_g, self.max_grad_norm)
        self.opt_g.step()

        # Backprop the discriminators
        self.opt_d.zero_grad()
        self.backward_d()
        util.clip_grad_norm(self.opt_d, self.max_grad_norm)
        self.opt_d.step()

    def get_loss_dict(self):
        """Get a dictionary of current errors for the model."""
        loss_dict = {
            # Generator loss
            'loss_gan': self.loss_gan,
            'loss_jc': self.loss_jc,
            'loss_mle': self.loss_mle,
            'loss_g': self.loss_g,

            # Discriminator loss
            'loss_d_src': self.loss_d_src,
            'loss_d_tgt': self.loss_d_tgt,
            'loss_d': self.loss_d
        }

        # Map scalars to floats for interpretation outside of the model
        loss_dict = {k: v.item() for k, v in loss_dict.items()
                     if isinstance(v, torch.Tensor)}

        return loss_dict

    def get_image_dict(self):
        """Get a dictionary of current images (src, tgt_real, tgt_fake) for the model.

        Returns: Dictionary containing numpy arrays of shape (batch_size, num_channels, height, width).
        Keys: {src, src2tgt, tgt2src}.
        """
        image_tensor_dict = {'src': self.src,
                             'src2tgt': self.src2tgt}

        if self.is_training:
            # When training, include full cycles
            image_tensor_dict.update({
                'tgt': self.tgt,
                'tgt2src': self.tgt2src
            })

        image_dict = {k: util.un_normalize(v) for k, v in image_tensor_dict.items()}

        return image_dict

    def on_epoch_end(self):
        """Callback for end of epoch.

        Update the learning rate by stepping the LR schedulers.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        """Get the current learning rate"""
        return self.optimizers[0].param_groups[0]['lr']

    def _data_parallel(self):
        self.g_src = nn.DataParallel(self.g_src, self.gpu_ids).to(self.device)
        self.g_tgt = nn.DataParallel(self.g_tgt, self.gpu_ids).to(self.device)
        if self.is_training:
            self.d_src = nn.DataParallel(self.d_src, self.gpu_ids).to(self.device)
            self.d_tgt = nn.DataParallel(self.d_tgt, self.gpu_ids).to(self.device)

    def _jc_preprocess(self):
        """Pre-process inputs for Jacobian Clamping. Doubles batch size.

        See Also:
            Algorithm 1 from https://arxiv.org/1802.08768v2
        """
        delta = torch.randn_like(self.src)
        src_jc = self.src + delta / delta.norm()
        src_jc.clamp_(-1, 1)
        self.src = torch.cat((self.src, src_jc), dim=0)

        delta = torch.randn_like(self.tgt)
        tgt_jc = self.tgt + delta / delta.norm()
        tgt_jc.clamp_(-1, 1)
        self.tgt = torch.cat((self.tgt, tgt_jc), dim=0)

    def _jc_postprocess(self):
        """Post-process outputs after Jacobian Clamping.

        Chunks `self.src` into `self.src` and `self.src_jc`,
        `self.src2tgt` into `self.src2tgt` and `self.src2tgt_jc`,
        and similarly for `self.tgt` and `self.tgt2src`.

        See Also:
            Algorithm 1 from https://arxiv.org/1802.08768v2
        """
        self.src, self.src_jc = self.src.chunk(2, dim=0)
        self.tgt, self.tgt_jc = self.tgt.chunk(2, dim=0)

        self.src2tgt, self.src2tgt_jc = self.src2tgt.chunk(2, dim=0)
        self.tgt2src, self.tgt2src_jc = self.tgt2src.chunk(2, dim=0)
