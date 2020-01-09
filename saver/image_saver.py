import numpy as np
import os

from datetime import datetime
from PIL import Image
from tensorboardX import SummaryWriter
from util import make_grid, un_normalize


class ImageSaver(object):
    """Saver for logging images during testing.

    Set `saver = ImageSaver(...)`, pass `saver.save` as the `batch_hook`
    argument to `evaluate`. Then every batch will get saved.

    Args:
        save_format (str): One of 'tensorboard' or 'disk'.
        save_dir (str): Directory for saving output images (disk only).
        name (str): Experiment name for saving to disk or TensorBoard.
        phase (str): One of 'train', 'val', or 'test'.
    """
    def __init__(self, save_format, save_dir, name, phase):
        self.phase = phase
        if save_format == 'tensorboard':
            log_dir = 'logs/{}_{}_{}'\
                .format(name, phase, datetime.now().strftime('%b%d_%H%M'))
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        elif save_format == 'disk':
            self.summary_writer = None
            self.save_dir = os.path.join(save_dir, 'images')
            os.makedirs(self.save_dir, exist_ok=True)
            for subdir in ('src2tgt', 'tgt2src'):
                os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
        else:
            raise ValueError('Invalid save format: {}'.format(save_format))
        self.global_step = 1

    def save(self, batch, src2tgt, tgt2src):
        # Un-normalize
        src, src2tgt = un_normalize(batch['src']), un_normalize(src2tgt)
        tgt, tgt2src = un_normalize(batch['tgt']), un_normalize(tgt2src)

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

    def save_labeled(self, batch, src2tgt, tgt2src):
        """Save a batch of labelled examples."""
        src, src2tgt = un_normalize(batch['src']), un_normalize(src2tgt)
        src_paths = batch['src_path']

        tgt, tgt2src = un_normalize(batch['tgt']), un_normalize(tgt2src)
        tgt_paths = batch['tgt_path']

        # Make grid of images
        i = 0
        for s, s2t, sp, t, t2s, tp in zip(src, src2tgt, src_paths,
                                          tgt, tgt2src, tgt_paths):
            # Save image
            if self.summary_writer is None:
                self._write(s2t, os.path.basename(sp), subdir='src2tgt')
                self._write(t2s, os.path.basename(tp), subdir='tgt2src')
            else:
                images_concat = make_grid([s, s2t], nrow=2, padding=4, pad_value=(243, 124, 42))
                self.summary_writer.add_image('src/src_src2tgt_{}'.format(i), images_concat, self.global_step)

                images_concat = make_grid([t, t2s], nrow=2, padding=4, pad_value=(243, 124, 42))
                self.summary_writer.add_image('tgt/tgt_tgt2src_{}'.format(i), images_concat, self.global_step)

            i += 1
            self.global_step += 1

    def _write(self, img, img_name, subdir=None):
        if subdir is not None:
            img_path = os.path.join(self.save_dir, subdir, img_name)
        else:
            img_path = os.path.join(self.save_dir, img_name)
        img_np = np.transpose(img.numpy(), [1, 2, 0])
        img = Image.fromarray(img_np)
        img.save(img_path)
