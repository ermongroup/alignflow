import torch.nn as nn


class ResNetBlock(nn.Module):
    """Block in ResNet.

    See Also:
        https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self, dim, norm_layer, use_bias, padding_type='reflect', use_dropout=False):
        super(ResNetBlock, self).__init__()

        if padding_type != 'reflect':
            raise NotImplementedError('Unsupported padding type: {}'.format(padding_type))

        layers = [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                  norm_layer(dim),
                  nn.ReLU(True)]

        if use_dropout:
            layers += [nn.Dropout(0.5)]

        layers += [nn.ReflectionPad2d(1),
                   nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                   norm_layer(dim)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input_):
        """Add residual connection to the conv block's output."""
        output = input_ + self.conv_block(input_)
        return output
