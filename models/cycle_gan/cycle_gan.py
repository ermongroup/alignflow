import os
import torch
import torch.nn as nn
import util

from itertools import chain
from models.patch_gan import PatchGAN
from models.cycle_gan.resnet import ResNet


class CycleGAN(nn.Module):
    """Cycle GAN model for image-to-image translation with cycle-consistency loss.

    Based on the paper:
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
    by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
    (https://arxiv.org/abs/1703.10593).
    """
    def __init__(self, args):
        """
        Args:
            args: Configuration args passed in via the command line.
        """
        super(CycleGAN, self).__init__()
        self.device = 'cuda' if len(args.gpu_ids) > 0 else 'cpu'
        self.gpu_ids = args.gpu_ids
        self.is_training = args.is_training

        # Set up generators
        self.g_src = ResNet(args)  # Maps from src to tgt
        self.g_tgt = ResNet(args)  # Maps from tgt to src

        if self.is_training:
            # Set up discriminators
            self.d_tgt = PatchGAN(args)  # Answers Q "is this tgt image real?"
            self.d_src = PatchGAN(args)  # Answers Q "is this src image real?"

            self._data_parallel()

            # Set up loss functions
            self.lambda_src = args.lambda_src  # Weight ratio of loss CYC_SRC:GAN
            self.lambda_tgt = args.lambda_tgt  # Weight ratio of loss CYC_TGT:GAN
            self.lambda_id = args.lambda_id    # Weight ratio of loss ID_{SRC,TGT}:CYC_{SRC,TGT}
            self.l1_loss_fn = nn.L1Loss()
            self.gan_loss_fn = util.GANLoss(device=self.device, use_least_squares=True)

            # Set up optimizers
            self.opt_g = torch.optim.Adam(chain(self.g_src.parameters(), self.g_tgt.parameters()),
                                          lr=args.lr,
                                          betas=(args.beta_1, args.beta_2))
            self.opt_d = torch.optim.Adam(chain(self.d_tgt.parameters(), self.d_src.parameters()),
                                          lr=args.lr,
                                          betas=(args.beta_1, args.beta_2))
            self.optimizers = [self.opt_g, self.opt_d]
            self.schedulers = [util.get_lr_scheduler(opt, args) for opt in self.optimizers]

            # Setup image mixers
            buffer_capacity = 50 if args.use_mixer else 0
            self.src2tgt_buffer = util.ImageBuffer(buffer_capacity)  # Buffer of generated tgt images
            self.tgt2src_buffer = util.ImageBuffer(buffer_capacity)  # Buffer of generated src images

            if args.clamp_jacobian:
                raise NotImplementedError('Jacobian Clamping not implemented for CycleGAN.')
        else:
            self._data_parallel()

        # Images in cycle src -> tgt -> src
        self.src = None
        self.src2tgt = None
        self.src2tgt2src = None

        # Images in cycle tgt -> src -> tgt
        self.tgt = None
        self.tgt2src = None
        self.tgt2src2tgt = None

        # Discriminator loss
        self.loss_d_tgt = None
        self.loss_d_src = None
        self.loss_d = None

        # Generator GAN loss
        self.loss_gan_src = None
        self.loss_gan_tgt = None
        self.loss_gan = None

        # Generator Identity loss
        self.src2src = None
        self.tgt2tgt = None
        self.loss_id_src = None
        self.loss_id_tgt = None
        self.loss_id = None

        # Generator Cycle loss
        self.loss_cyc_src = None
        self.loss_cyc_tgt = None
        self.loss_cyc = None

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
            self.src2tgt = self.g_src(self.src)
            self.tgt2src = self.g_tgt(self.tgt)

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
        # Forward src -> tgt: Say target is real to invert loss
        self.src2tgt = self.g_src(self.src)
        self.loss_gan_src = self.gan_loss_fn(self.d_tgt(self.src2tgt), is_tgt_real=True)

        # Forward tgt -> src: Say target is real to invert loss
        self.tgt2src = self.g_tgt(self.tgt)
        self.loss_gan_tgt = self.gan_loss_fn(self.d_src(self.tgt2src), is_tgt_real=True)

        # Cycle: src -> tgt -> src
        self.src2tgt2src = self.g_tgt(self.src2tgt)
        self.loss_cyc_src = self.lambda_src * self.l1_loss_fn(self.src2tgt2src, self.src)

        # Cycle: tgt -> src -> tgt
        self.tgt2src2tgt = self.g_src(self.tgt2src)
        self.loss_cyc_tgt = self.lambda_tgt * self.l1_loss_fn(self.tgt2src2tgt, self.tgt)

        # Identity: src -> src
        self.src2src = self.g_tgt(self.src)
        self.loss_id_src = self.lambda_id * self.lambda_src * self.l1_loss_fn(self.src2src, self.src)

        # Identity: tgt -> tgt
        self.tgt2tgt = self.g_src(self.tgt)
        self.loss_id_tgt = self.lambda_id * self.lambda_tgt * self.l1_loss_fn(self.tgt2tgt, self.tgt)

        # Total losses
        self.loss_gan = self.loss_gan_src + self.loss_gan_tgt
        self.loss_cyc = self.loss_cyc_src + self.loss_cyc_tgt
        self.loss_id = self.loss_id_src + self.loss_id_tgt

        # Backprop
        self.loss_g = self.loss_gan + self.loss_cyc + self.loss_id
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
        self.opt_g.step()

        # Backprop the discriminators
        self.opt_d.zero_grad()
        self.backward_d()
        self.opt_d.step()

    def get_loss_dict(self):
        """Get a dictionary of current errors for the model."""
        loss_dict = {
            # Generator loss
            'loss_gan': self.loss_gan,
            'loss_cyc': self.loss_cyc,
            'loss_id': self.loss_id,
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
        image_tensor_dict = {'src': self.src, 'src2tgt': self.src2tgt}

        if self.is_training:
            # When training, include full cycles
            image_tensor_dict.update({
                'src2tgt2src': self.src2tgt2src,
                'tgt': self.tgt,
                'tgt2src': self.tgt2src,
                'tgt2src2tgt': self.tgt2src2tgt
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
