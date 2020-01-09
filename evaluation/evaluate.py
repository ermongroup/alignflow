import time
import torch

from tqdm import tqdm
from util import AverageMeter


def evaluate(model, data_loader, criteria, max_examples=int(1e10), batch_hook=None):
    """Evaluate a model.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Loader for data to evaluate on.
        criteria (dict): Dictionary mapping strings to functions, where each
            criterion function takes outputs and targets, and produces a number.
        max_examples (int): Maximum number of examples on which to evaluate.
        batch_hook (func): Callback to call with (src, src2tgt, tgt, tgt2src) images
            after each batch.

    Returns:
        Dictionary mapping strings (one per criterion) to the average value
            returned by that criterion on the dataset.
    """
    time_meter = AverageMeter()
    meters = {k: AverageMeter() for k in criteria}

    num_examples = min(len(data_loader.dataset), max_examples)
    with tqdm(total=num_examples) as progress_bar:
        with torch.no_grad():
            for batch in data_loader:
                start = time.time()
                batch_size = batch['src'].size(0)

                # Evaluate one src -> tgt batch
                model.set_inputs(batch['src'], batch['tgt'])
                model.test()
                for criterion_name, criterion_fn in criteria.items():
                    if criterion_name.endswith('tgt2src'):
                        criterion_val = criterion_fn(model.tgt2src, model.src).item()
                    else:
                        # Assume forward direction
                        criterion_val = criterion_fn(model.src2tgt, model.tgt).item()
                    criterion_meter = meters[criterion_name]
                    criterion_meter.update(criterion_val, batch_size)
                time_meter.update(time.time() - start, batch_size)

                if batch_hook is not None:
                    batch_hook(batch['src'], model.src2tgt, batch['tgt'], model.tgt2src)

                progress_bar.set_postfix(time=time_meter.avg,
                                         **{k: v.avg for k, v in meters.items()})
                progress_bar.update(batch_size)

    stats = {k: v.avg for k, v in meters.items()}

    return stats
