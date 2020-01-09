import torch.nn as nn


def init_model(model, init_method='normal'):
    """Initialize model parameters.

    Args:
        model: Model to initialize.
        init_method: Name of initialization method: 'normal' or 'xavier'.
    """
    # Initialize model parameters
    if init_method == 'normal':
        model.apply(_normal_init)
    elif init_method == 'xavier':
        model.apply(_xavier_init)
    else:
        raise NotImplementedError('Invalid weights initializer: {}'.format(init_method))


def _normal_init(model):
    """Apply normal initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('Linear') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def _xavier_init(model):
    """Apply Xavier initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find('Linear') != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)
