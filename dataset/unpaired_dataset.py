import os
import random

from dataset.base_dataset import BaseDataset
from PIL import Image


class UnpairedDataset(BaseDataset):
    """Dataset of unpaired images from two domains.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
        shuffle_pairs (bool): Shuffle the pairs so that the image from domain B that appears
            with a given image from domain A is random.
        resize_shape (tuple or list): Side lengths for resizing images.
        crop_shape (tuple or list): Side lengths for cropping images.
        direction (str): One of 'ab' or 'ba'.
    """
    def __init__(self, data_dir, phase, shuffle_pairs, resize_shape, crop_shape, direction='ab'):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        if direction not in ('ab', 'ba'):
            raise ValueError('Invalid direction: {}'.format(direction))

        super(UnpairedDataset, self).__init__(data_dir, phase, resize_shape, crop_shape)
        self.a_dir = os.path.join(data_dir, phase + 'A')
        self.b_dir = os.path.join(data_dir, phase + 'B')
        self.a_paths = sorted(self.get_image_paths(self.a_dir))
        self.b_paths = sorted(self.get_image_paths(self.b_dir))
        if shuffle_pairs:
            random.shuffle(self.b_paths)
        self.reverse = (direction == 'ba')
        self.shuffle_pairs = shuffle_pairs
        self.transform_fn = self._get_transform_fn()

    def __getitem__(self, index):
        a_path = self.a_paths[index % len(self.a_paths)]
        b_path = self.b_paths[index % len(self.b_paths)]

        a_img = Image.open(a_path).convert('RGB')
        b_img = Image.open(b_path).convert('RGB')

        a_img = self.transform_fn(a_img)
        b_img = self.transform_fn(b_img)

        return {'src': b_img if self.reverse else a_img,
                'src_path': b_path if self.reverse else a_path,
                'tgt': a_img if self.reverse else b_img,
                'tgt_path': a_path if self.reverse else b_path}

    def __len__(self):
        return max(len(self.a_paths), len(self.b_paths))
