import models

from args import TrainArgParser
from dataset import PairedDataset, UnpairedDataset
from evaluation import evaluate
from evaluation.criteria import mse
from logger import TrainLogger
from saver import ModelSaver
from torch.utils.data import DataLoader


def train(args):
    # Get model
    model = models.__dict__[args.model](args)
    if args.ckpt_path:
        model = ModelSaver.load_model(model, args.ckpt_path, args.gpu_ids, is_training=True)
    model = model.to(args.device)
    model.train()

    # Get loader, logger, and saver
    train_loader, val_loader = get_data_loaders(args)
    logger = TrainLogger(args, model, dataset_len=len(train_loader.dataset))
    saver = ModelSaver(args.save_dir, args.max_ckpts, metric_name=args.metric_name,
                       maximize_metric=args.maximize_metric, keep_topk=True)

    # Train
    while not logger.is_finished_training():
        logger.start_epoch()
        for batch in train_loader:
            logger.start_iter()

            # Train over one batch
            model.set_inputs(batch['src'], batch['tgt'])
            model.train_iter()

            logger.end_iter()

            # Evaluate
            if logger.global_step % args.iters_per_eval < args.batch_size:
                criteria = {'MSE_src2tgt': mse, 'MSE_tgt2src': mse}
                stats = evaluate(model, val_loader, criteria)
                logger.log_scalars({'val_' + k: v for k, v in stats.items()})
                saver.save(logger.global_step, model,
                           stats[args.metric_name], args.device)

        logger.end_epoch()


def get_data_loaders(args):
    train_dataset = UnpairedDataset(args.data_dir,
                                    phase='train',
                                    shuffle_pairs=True,
                                    resize_shape=args.resize_shape,
                                    crop_shape=args.crop_shape,
                                    direction=args.direction)
    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    val_dataset = PairedDataset(args.data_dir,
                                phase='val',
                                resize_shape=args.resize_shape,
                                crop_shape=args.crop_shape,
                                direction=args.direction)
    val_loader = DataLoader(val_dataset,
                            args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=True)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
