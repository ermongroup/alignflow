import os
import queue
import shutil
import torch
import torch.nn as nn


class ModelSaver(object):
    """Class to save and load model checkpoints.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_ckpts (int): Maximum number of checkpoints to keep before overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes the metric value passed in via save.
            If false, best checkpoint minimizes the metric.
        keep_topk (bool): Keep the top K checkpoints, rather than the most recent K checkpoints.
    """
    def __init__(self, save_dir, max_ckpts, metric_name, maximize_metric=False, keep_topk=True):
        super(ModelSaver, self).__init__()

        self.save_dir = save_dir
        self.max_ckpts = max_ckpts
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_metric_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.keep_topk = keep_topk

    def _is_best(self, metric_val):
        """Check whether `metric_val` is the best one we've seen so far.

        Args:
            metric_val (float): Metric value to compare to all previous checkpoints.
        """
        if metric_val is None:
            return False
        return (self.best_metric_val is None
                or (self.maximize_metric and self.best_metric_val < metric_val)
                or (not self.maximize_metric and self.best_metric_val > metric_val))

    def save(self, iteration, model, metric_val, device):
        """If this iteration corresponds to a save iteration, save model parameters to disk.

        Args:
            iteration (int): Iteration that just finished.
            model (nn.Module): Model to save.
            metric_val (float): Value for determining whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.to('cpu').state_dict(),
            'iteration': iteration
        }
        model.to(device)

        ckpt_path = os.path.join(self.save_dir, 'iter_{}.pth.tar'.format(iteration))
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_val):
            # Save the best model
            self.best_metric_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(ckpt_path, best_path)

        # Add checkpoint path to priority queue (lower priority order gets removed first)
        if not self.keep_topk:
            priority_order = iteration
        elif self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, ckpt_path))

        # Remove a checkpoint if more than max_ckpts ckpts saved
        if self.ckpt_paths.qsize() > self.max_ckpts:
            _, oldest_ckpt = self.ckpt_paths.get()
            try:
                os.remove(oldest_ckpt)
            except OSError:
                pass

    @classmethod
    def load_model(cls, model, ckpt_path, gpu_ids, is_training=False):
        """Load model parameters from disk.

        Args:
            model (nn.DataParallel): Uninitialized model to load parameters into.
            ckpt_path (str): Path to checkpoint to load.
            gpu_ids (list): GPU IDs for DataParallel.
            is_training (bool): Whether training (if False, ignore
                `d_src.*` and `d_tgt.*` parameters).

        Returns:
            Model loaded from checkpoint, dict of additional checkpoint info (e.g. epoch, metric).
        """
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)

        if is_training:
            state_dict = ckpt_dict['model_state']
        else:
            state_dict = {k: v for k, v in ckpt_dict['model_state'].items()
                          if not (k.startswith('d_src') or k.startswith('d_tgt'))}

        # Build model, load parameters
        model.load_state_dict(state_dict)

        return model
