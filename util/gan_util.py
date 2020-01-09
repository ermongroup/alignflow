import random
import torch
import torch.nn as nn

from argparse import ArgumentError


def gan_class_name(s):
    """Convert a string to a case-sensitive GAN class name.
    Used to parse gan name from the command line
    """
    s = s.lower().strip().replace('_', '')
    if s == 'pix2pix':
        class_name = 'Pix2Pix'
    elif s == 'flowpix2pix':
        class_name = 'FlowPix2Pix'
    elif s == 'cyclegan':
        class_name = 'CycleGAN'
    elif s == 'cycleflow':
        class_name = 'CycleFlow'
    elif s == 'flow2flow':
        class_name = 'Flow2Flow'
    else:
        raise ArgumentError('Argument does not match a GAN type: {}'.format(s))

    return class_name


class GANLoss(nn.Module):
    """Module for computing the GAN loss for the generator.

    When `use_least_squares` is turned on, we use mean squared error loss,
    otherwise we use the standard binary cross-entropy loss.

    Note: We use the convention that the discriminator predicts the probability
    that the target image is real. Therefore real corresponds to label 1.0."""
    def __init__(self, device, use_least_squares=False):
        super(GANLoss, self).__init__()
        self.loss_fn = nn.MSELoss() if use_least_squares else nn.BCELoss()
        self.real_label = None  # Label tensor passed to loss if target is real
        self.fake_label = None  # Label tensor passed to loss if target is fake
        self.device = device

    def _get_label_tensor(self, input_, is_tgt_real):
        # Create label tensor if needed
        if is_tgt_real and (self.real_label is None or self.real_label.numel() != input_.numel()):
            self.real_label = torch.ones_like(input_, device=self.device, requires_grad=False)
        elif not is_tgt_real and (self.fake_label is None or self.fake_label.numel() != input_.numel()):
            self.fake_label = torch.zeros_like(input_, device=self.device, requires_grad=False)

        return self.real_label if is_tgt_real else self.fake_label

    def __call__(self, input_, is_tgt_real):
        label = self._get_label_tensor(input_, is_tgt_real)
        return self.loss_fn(input_, label)

    def forward(self, input_):
        raise NotImplementedError('GANLoss should be called directly.')


class ImageBuffer(object):
    """Holds a buffer of old generated images for training. Stabilizes training
    by allowing us to feed a history of generated examples to the discriminator,
    so the discriminator cannot just focus on the newest examples.

    Based on ideas from Section 2.3 of the paper:
    "Learning from Simulated and Unsupervised Images through Adversarial Training"
    by Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Josh Susskind, Wenda Wang, Russ Webb
    (http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf)
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: Size of the pool for mixing. Set to 0 to disable image mixer.
        """
        self.capacity = capacity
        self.buffer = []

    def sample(self, images):
        """Sample old images and mix new images into the buffer.

        Args:
            images: New example images to mix into the buffer.

        Returns:
            Tensor batch of images that are mixed from the buffer.
        """
        if self.capacity == 0:
            return images

        # Add to the pool
        mixed_images = []  # Mixture of pool and input images
        for new_img in images:
            new_img = torch.unsqueeze(new_img.data, 0)

            if len(self.buffer) < self.capacity:
                # Pool is still filling, so always add
                self.buffer.append(new_img)
                mixed_images.append(new_img)
            else:
                # Pool is filled, so mix into the pool with probability 1/2
                if random.uniform(0, 1) < 0.5:
                    mixed_images.append(new_img)
                else:
                    pool_img_idx = random.randint(0, len(self.buffer) - 1)
                    mixed_images.append(self.buffer[pool_img_idx].clone())
                    self.buffer[pool_img_idx] = new_img

        return torch.cat(mixed_images, 0)


class JacobianClampingLoss(nn.Module):
    """Module for adding Jacobian Clamping loss.

    See Also:
        https://arxiv.org/abs/1802.08768v2
    """
    def __init__(self, lambda_min=1., lambda_max=20.):
        super(JacobianClampingLoss, self).__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, gz, gz_prime, z, z_prime):
        q = (gz - gz_prime).norm() / (z - z_prime).norm()
        l_max = (q.clamp(self.lambda_max, float('inf')) - self.lambda_max) ** 2
        l_min = (q.clamp(float('-inf'), self.lambda_min) - self.lambda_min) ** 2
        l_jc = l_max + l_min

        return l_jc
