import torch.nn as nn

from util import init_model, get_norm_layer


class PatchGAN(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, args):
        """Constructs a basic PatchGAN convolutional discriminator.

        Each position in the output is a score of discriminator confidence that
        a 70x70 patch of the input is real.

        Args:
            args: Arguments passed in via the command line.
        """
        super(PatchGAN, self).__init__()

        norm_layer = get_norm_layer(args.norm_type)

        layers = []

        # Double channels for conditional GAN (concatenated src and tgt images)
        num_channels = args.num_channels

        layers += [nn.Conv2d(num_channels, args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(args.num_channels_d, 2 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(2 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(2 * args.num_channels_d, 4 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(4 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(4 * args.num_channels_d, 8 * args.num_channels_d, args.kernel_size_d, stride=1, padding=1),
                   norm_layer(8 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(8 * args.num_channels_d, 1, args.kernel_size_d, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)
        init_model(self.model, init_method=args.initializer)

    def forward(self, input_):
        return self.model(input_)
