import json
import models
import numpy as np
import os
import shutil
import torch.nn as nn

from args import TestArgParser
from dataset import PairedDataset
from datetime import datetime
from evaluation import evaluate
from evaluation.criteria import mse
from PIL import Image
from saver import ModelSaver
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util import make_grid, un_normalize


def test(args):
    # Get dataset
    dataset = PairedDataset(args.data_dir,
                            phase=args.phase,
                            resize_shape=args.resize_shape,
                            crop_shape=args.crop_shape,
                            direction=args.direction)
    data_loader = DataLoader(dataset,
                             args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    # Get model
    model = models.__dict__[args.model](args)
    model = ModelSaver.load_model(model, args.ckpt_path, args.gpu_ids)
    model.train()

    # Set up image saving
    if args.save_images is None:
        save_hook = None
    else:
        saver = ImageSaver(args.save_images, args.results_dir, args.name, args.phase)
        save_hook = saver.save

    # Test model
    criteria = {'MSE_src2tgt': mse, 'MSE_tgt2src': mse}
    stats = evaluate(model, data_loader, criteria, batch_hook=save_hook)

    # Add model info to stats
    stats.update({
        'name': args.name,
        'ckpt_path': args.ckpt_path
    })

    # Write stats to disk
    stats_path = os.path.join(args.results_dir, 'stats.json')
    print('Saving stats to {}...'.format(stats_path))
    with open(stats_path, 'w') as json_fh:
        json.dump(stats, json_fh, sort_keys=True, indent=4)

    # Copy training args for reference
    args_src = os.path.join(args.save_dir, 'args.json')
    args_dst = os.path.join(args.results_dir, 'args.json')
    print('Copying args to {}...'.format(args_dst))
    shutil.copy(args_src, args_dst)


class ImageSaver(object):
    """Saver for logging images during testing.

    Set `saver = ImageSaver(...)`, pass `saver.save` as the `batch_hook`
    argument to `evaluate`. Then every batch will get saved.

    Args:
        save_format (str): One of 'tensorboard' or 'disk'.
        results_dir (str): Directory for saving output images (disk only).
        name (str): Experiment name for saving to disk or TensorBoard.
        phase (str): One of 'train', 'val', or 'test'.
    """
    def __init__(self, save_format, results_dir, name, phase):
        self.phase = phase
        if save_format == 'tensorboard':
            log_dir = 'logs/{}_{}_{}'\
                .format(name, phase, datetime.now().strftime('%b%d_%H%M'))
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        elif save_format == 'disk':
            self.summary_writer = None
            self.save_dir = os.path.join(results_dir, 'images')
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            raise ValueError('Invalid save format: {}'.format(save_format))
        self.global_step = 1

    def save(self, src, src2tgt, tgt, tgt2src):
        # Un-normalize
        src, src2tgt = un_normalize(src), un_normalize(src2tgt)
        tgt, tgt2src = un_normalize(tgt), un_normalize(tgt2src)

        # Make grid of images
        i = 0
        for s, s2t, t, t2s in zip(src, src2tgt, tgt, tgt2src):
            # Save image
            if self.summary_writer is None:
                images_concat = make_grid([s, s2t], nrow=2, padding=0)
                file_name = 'src2tgt_{:04d}.png'.format(self.global_step)
                self._write(images_concat, file_name)

                images_concat = make_grid([t, t2s], nrow=2, padding=0)
                file_name = 'tgt2src_{:04d}.png'.format(self.global_step)
                self._write(images_concat, file_name)
            else:
                images_concat = make_grid([s, s2t], nrow=2, padding=4, pad_value=(243, 124, 42))
                self.summary_writer.add_image('src/src_src2tgt_{}'.format(i), images_concat, self.global_step)

                images_concat = make_grid([t, t2s], nrow=2, padding=4, pad_value=(243, 124, 42))
                self.summary_writer.add_image('tgt/tgt_tgt2src_{}'.format(i), images_concat, self.global_step)

            i += 1
            self.global_step += 1

    def _write(self, img, img_name):
        img_path = os.path.join(self.save_dir, img_name)
        img_np = np.transpose(img.numpy(), [1, 2, 0])
        img = Image.fromarray(img_np)
        img.save(img_path)


if __name__ == '__main__':
    parser = TestArgParser()
    test(parser.parse_args())
