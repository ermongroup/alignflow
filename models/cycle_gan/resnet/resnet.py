import torch.nn as nn

from models.cycle_gan.resnet.resnet_block import ResNetBlock
from util import get_norm_layer, init_model


class ResNet(nn.Module):
    """ResNet model for the generator.

    Down-sample, feed through 9 ResNet blocks, then up-sample.

    See Also:
        https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self, args):
        super(ResNet, self).__init__()
        num_blocks = 9
        num_channels = args.num_channels
        num_filters = args.num_channels_g
        norm_layer = get_norm_layer(args.norm_type)
        use_bias = args.norm_type == 'instance'

        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(num_channels, num_filters, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(num_filters),
                  nn.ReLU(inplace=True)]

        num_down = 2  # Number of times to down-sample
        for i in range(num_down):
            k = 2**i
            layers += [nn.Conv2d(k * num_filters, 2*k*num_filters, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(2 * k * num_filters),
                       nn.ReLU(inplace=True)]

        k = 2**num_down
        layers += [ResNetBlock(k * num_filters, norm_layer, use_bias=use_bias) for _ in range(num_blocks)]

        for i in range(num_down):
            k = 2**(num_down - i)
            layers += [nn.ConvTranspose2d(k * num_filters, int(k * num_filters / 2), kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=use_bias),
                       norm_layer(int(k * num_filters / 2)),
                       nn.ReLU(inplace=True)]

        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(num_filters, num_channels, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model = nn.Sequential(*layers)
        init_model(self.model, init_method=args.initializer)

    def forward(self, input_):
        return self.model(input_)
