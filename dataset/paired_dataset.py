import os

from dataset.base_dataset import BaseDataset
from PIL import Image


class PairedDataset(BaseDataset):
    """Dataset of paired images from two domains.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
        resize_shape (tuple or list): Side lengths for resizing images.
        crop_shape (tuple or list): Side lengths for cropping images.
        direction (str): One of 'ab' or 'ba'.
    """
    def __init__(self, data_dir, phase, resize_shape, crop_shape, direction):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        if direction not in ('ab', 'ba'):
            raise ValueError('Invalid direction: {}'.format(direction))

        super(PairedDataset, self).__init__(data_dir, phase, resize_shape, crop_shape)
        self.ab_dir = os.path.join(self.data_dir, phase)
        self.ab_paths = sorted(self.get_image_paths(self.ab_dir))
        self.transform_fn = self._get_transform_fn()
        self.reverse = (direction == 'ba')

    def __getitem__(self, index):
        ab_path = self.ab_paths[index]

        ab_img = Image.open(ab_path).convert('RGB')

        # Split image
        w, h = ab_img.size
        w2 = int(w / 2)
        a_img = ab_img.crop((0, 0, w2, h))
        b_img = ab_img.crop((w2, 0, w, h))

        a_img = self.transform_fn(a_img)
        b_img = self.transform_fn(b_img)

        return {'src': b_img if self.reverse else a_img,
                'src_path': ab_path,
                'tgt': a_img if self.reverse else b_img,
                'tgt_path': ab_path}

    def __len__(self):
        return len(self.ab_paths)
