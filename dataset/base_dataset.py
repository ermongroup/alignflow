import os
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


class BaseDataset(data.Dataset):
    """Base dataset of images from two domains, subclassed by `PairedDataset`
    and `UnpairedDataset`.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
        resize_shape (tuple or list): Side lengths for resizing images.
        crop_shape (tuple or list): Side lengths for cropping images.
    """
    def __init__(self, data_dir, phase, resize_shape=(286, 286), crop_shape=(256, 256)):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))

        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.phase = phase
        self.transform_fn = self._get_transform_fn()

    def _get_transform_fn(self):
        transforms_list = []
        if self.resize_shape is not None:
            transforms_list.append(transforms.Resize(self.resize_shape, Image.BICUBIC))
        if self.crop_shape is not None:
            if self.phase == 'train':
                transforms_list.append(transforms.RandomCrop(self.crop_shape))
            else:
                transforms_list.append(transforms.CenterCrop(self.crop_shape))
        transforms_list += [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transforms_list)

    @staticmethod
    def _is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_image_paths(self, data_dir):
        """Get paths to images in a folder of images."""
        images = []
        assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir

        for root, _, filenames in sorted(os.walk(data_dir)):
            for filename in filenames:
                if self._is_image_file(filename):
                    path = os.path.join(root, filename)
                    images.append(path)

        return images

    def __getitem__(self, idx):
        raise NotImplementedError('Subclass of BaseDataset must implement __getitem__')

    def __len__(self):
        raise NotImplementedError('Subclass of BaseDataset must implement __len__')
