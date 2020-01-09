import functools
import torch
import torch.nn as nn


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        return functools.partial(nn.GroupNorm, num_groups=16)
    else:
        raise NotImplementedError('Invalid normalization type: {}'.format(norm_type))


def get_param_groups(net, weight_decay, norm_suffix='weight_g', verbose=False):
    """Get two parameter groups from `net`: One named "normalized" which will
    override the optimizer with `weight_decay`, and one named "unnormalized"
    which will inherit all hyperparameters from the optimizer.

    Args:
        net (torch.nn.Module): Network to get parameters from
        weight_decay (float): Weight decay to apply to normalized weights.
        norm_suffix (str): Suffix to select weights that should be normalized.
            For WeightNorm, using 'weight_g' normalizes the scale variables.
        verbose (bool): Print out number of normalized and unnormalized parameters.
    """
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x


class BatchNorm2dStats(nn.Module):
    """Compute BatchNorm2d normalization statistics: `mean` and `var`.

    Useful for keeping track of sum of log-determinant of Jacobians in flow models.

    Args:
        num_features (int): Number of features in the input (i.e., `C` in `(N, C, H, W)`).
        track_running_stats (bool): Track the running mean and variance during training.
        eps (float): Added to the denominator for numerical stability.
        momentum (float): The value used for the running_mean and running_var computation.
            Different from conventional momentum, see `nn.BatchNorm2d` for more.
    """
    def __init__(self, num_features, track_running_stats=True, eps=1e-5, momentum=0.1):
        super(BatchNorm2dStats, self).__init__()
        self.keep_running_stats = track_running_stats
        self.eps = eps

        if self.keep_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.momentum = momentum

    def forward(self, x):
        # Get mean and variance per channel
        if self.training or not self.keep_running_stats:
            channels = x.transpose(0, 1).contiguous().view(x.size(1), -1)
            mean = channels.mean(1)
            var = channels.var(1)

            if self.keep_running_stats:
                # Update variables
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        var += self.eps

        # Reshape to (N, C, H, W)
        mean = mean.view(1, x.size(1), 1, 1).expand_as(x)
        var = var.view(1, x.size(1), 1, 1).expand_as(x)

        return mean, var
