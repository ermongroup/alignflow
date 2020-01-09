import os

from imageio import imsave
from logger.base_logger import BaseLogger
from time import time


class TestLogger(BaseLogger):
    """Logs test info."""
    def __init__(self, args, model, dataset_len):
        assert not args.is_training
        super(TestLogger, self).__init__(args, model, dataset_len)

        self.batch_size = args.batch_size
        self.num_examples = args.num_examples
        self.src_paths = None

    def start_iter(self, src_paths=None):
        """Log info for start of an iteration."""
        self.iter_start_time = time()
        self.src_paths = src_paths

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

        # Write test examples to output directory
        image_dict = self.model.get_image_dict()
        image_keys = ('src', 'src2tgt')

        for k in image_keys:
            image_dir = os.path.join(self.save_dir, k)
            os.makedirs(image_dir, exist_ok=True)

            num_images = min(self.batch_size, len(image_dict[k]))
            for i in range(num_images):
                file_name = os.path.basename(self.src_paths[i])
                image_path = os.path.join(self.save_dir, k, file_name)
                image = image_dict[k][i].permute(1, 2, 0).cpu().numpy()
                imsave(image_path, image)

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of test: writing to {}]'.format(self.save_dir))

    def end_epoch(self):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        self.write('[end of test, time: {:.2f}]'.format(time() - self.epoch_start_time))
        self.epoch += 1

    def is_finished_testing(self):
        """Return True if max number of examples have been tested, else return False."""
        return 0 < self.num_examples < self.iter
