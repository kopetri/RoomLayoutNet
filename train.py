from pathlib import Path
from numpy.core.fromnumeric import take
import torch
import sys
import random
from argparse import ArgumentParser
from module import LayoutSegmentation
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import CornerDataset
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser('Train view quality model')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=50, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--find_learning_rate', action='store_true', help="Finding learning rate.")
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')

    parser.add_argument('--dataset_path', required=True, type=str, help="Path to datasets.")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size.")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate.")
    parser.add_argument('--learning_rate_decay', default=0.9999, type=float)
    parser.add_argument('--early_stop_patience', default=0, type=int, help='Stop training after n epochs with ne val_loss improvement.')
    parser.add_argument('--name', required=True, type=str, help="Name of the training run.")
    parser.add_argument('--loss', default='mse', type=str, help="Base loss function")

    # bts
    parser.add_argument('--encoder', type=str, help='type of encoder, desenet121_bts, densenet161_bts, resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts', default='densenet161_bts')
    parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512) 
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=1)                                                           

    args = parser.parse_args()
    args.dataset = Path(args.dataset_path).stem

    if args.detect_anomaly:
        print("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)
    
    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    callbacks = []

    if args.learning_rate_decay:
        callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]

    callbacks += [pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_top_k=5,
        filename='{epoch}-{valid_3DIoU}',
        monitor='valid_3DIoU',
        mode='max'
    )]

    if args.early_stop_patience > 0:
        callbacks += [pl.callbacks.EarlyStopping(
            monitor='valid_3DIoU',
            min_delta=0.00,
            patience=args.early_stop_patience,
            verbose=True,
            mode='max'
        )]

    use_gpu = not args.gpus == 0

    trainer = pl.Trainer(
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=None,
        gpus=args.gpus,
        log_every_n_steps=1,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.WandbLogger(project="Layoutestimation", name=args.name),
        callbacks=callbacks
        )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })

    model = LayoutSegmentation(args)
    train_dataset = CornerDataset(path=args.dataset_path, split="train", scale=1.0)
    val_dataset   = CornerDataset(path=args.dataset_path, split="valid", scale=1.0)
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True,
                              num_workers=args.worker,
                              pin_memory=True,
                              worker_init_fn=lambda x: np.random.seed())

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)