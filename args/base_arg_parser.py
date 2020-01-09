import argparse
import os
import json
import torch

from util import args_to_list, gan_class_name, print_err, str_to_bool


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=gan_class_name, required=True,
                                 choices=('CycleGAN', 'CycleFlow', 'Flow2Flow'),
                                 help='Name of model to run. Case-insensitive.')
        self.parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='ckpts/',
                                 help='Directory in which to save checkpoints.')
        self.parser.add_argument('--ckpt_path', type=str, default='',
                                 help='Path to model checkpoint to load.')
        self.parser.add_argument('--data_dir', type=str, required=True,
                                 help='Path to data directory.')
        self.parser.add_argument('--direction', type=lambda s: s.lower(), default='ab', choices=('ab', 'ba'),
                                 help='Direction of source to target mapping.')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--initializer', type=str, default='normal', choices=('xavier', 'he', 'normal'),
                                 help='Initializer to use for all network parameters.')
        self.parser.add_argument('--kernel_size_d', default=4, type=int, help='Size of the discriminator\'s kernels.')
        self.parser.add_argument('--name', type=str, required=True, help='Experiment name.')
        self.parser.add_argument('--norm_type', type=str, default='instance', choices=('instance', 'batch', 'group'),
                                 help='Normalization type.')
        self.parser.add_argument('--num_scales', default=2, type=int, help='Number of scales in Real NVP.')
        self.parser.add_argument('--num_blocks', default=8, type=int, help='Number of res. blocks in s/t (Real NVP).')
        self.parser.add_argument('--resize_shape', type=str, default='286,286',
                                 help='Comma-separated 2D shape for images after resizing (before cropping).\
                                      By default, no resizing is applied.')
        self.parser.add_argument('--crop_shape', type=str, default='256,256',
                                 help='Comma-separated 2D shape for images after cropping (crop comes after resize).\
                                      By default, no cropping is applied.')
        self.parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in an image.')
        self.parser.add_argument('--num_channels_d', default=64, type=int,
                                 help='Number of filters in the discriminator\'s first convolutional layer.')
        self.parser.add_argument('--num_channels_g', default=64, type=int,
                                 help='Number of filters in the generator\'s first convolutional layer.')
        self.parser.add_argument('--num_workers', default=16, type=int, help='Number of DataLoader worker threads.')
        self.parser.add_argument('--phase', default='train', type=str, help='One of "train", "val", or "test".')
        self.parser.add_argument('--use_dropout', default=False, type=str_to_bool,
                                 help='Use dropout in the generator.')
        self.is_training = None
        self.no_mixer = None
        self.use_dropout = None

    def parse_args(self):
        args = self.parser.parse_args()

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        if not args.is_training and not args.ckpt_path:
            raise ValueError('Must specify --ckpt_path in test mode.')

        # Set up resize and crop
        args.resize_shape = args_to_list(args.resize_shape, arg_type=int, allow_empty=False)
        args.crop_shape = args_to_list(args.crop_shape, arg_type=int, allow_empty=False)

        # Set up available GPUs
        args.gpu_ids = args_to_list(args.gpu_ids, arg_type=int, allow_empty=True)
        if len(args.gpu_ids) > 0:
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda:{}'.format(args.gpu_ids[0])
        else:
            args.device = 'cpu'

        # Set up save dir and output dir (test mode only)
        args.save_dir = os.path.join(args.checkpoints_dir, args.name)
        os.makedirs(args.save_dir, exist_ok=True)
        if self.is_training:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as fh:
                json.dump(vars(args), fh, indent=4, sort_keys=True)
                fh.write('\n')
        else:
            args.results_dir = os.path.join(args.results_dir, args.name)
            os.makedirs(args.results_dir, exist_ok=True)

        return args
