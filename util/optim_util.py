import numpy as np

from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler as torch_scheduler


def bits_per_dim(x, loss, k=256):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        loss (torch.Tensor): Scalar loss tensor.
        k (int): Number of possible values per entry.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = loss / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    if max_norm > 0:
        for group in optimizer.param_groups:
            clip_grad_norm_(group['params'], max_norm, norm_type)


def get_lr_scheduler(optimizer, args):
    """Get learning rate scheduler."""
    if args.lr_policy == 'step':
        scheduler = torch_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = torch_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'linear':
        # After `lr_warmup_epochs` epochs, decay linearly to 0 for `lr_decay_epochs` epochs.
        def get_lr_multiplier(epoch):
            init_epoch = 1
            return 1.0 - max(0, epoch + init_epoch - args.lr_warmup_epochs) / float(args.lr_decay_epochs + 1)
        scheduler = torch_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    else:
        return NotImplementedError('Invalid learning rate policy: {}'.format(args.lr_policy))
    return scheduler
