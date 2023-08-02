import argparse
import os
from pathlib import Path
from typing import Tuple

import pytorch_lightning
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from parameter_prediction_network.arbitrary_style_helpers import get_dataloader_device
from parameter_prediction_network.path_utils import get_exp_name_and_interm_folder
from parameter_prediction_network.ppn_architectures import PPNArchitectures

# use new stuff
torch.backends.cuda.matmul.allow_tf32 = True


def get_dataloaders(datset_path: str, batch_size: int, num_workers: int,
                    model: pytorch_lightning.LightningModule, args: argparse.Namespace) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform_list = model.get_dataloader_transforms(args)
    transform = transforms.Compose(transform_list)
    dataset = datasets.ImageFolder(datset_path, transform)
    num_train_samples = int(len(dataset) * 0.95)
    num_valid_samples = len(dataset) - num_train_samples
    train, valid = random_split(dataset, [num_train_samples, num_valid_samples])
    multiprocessing_method = 'spawn' if 'cuda' in str(get_dataloader_device(args)) else None
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, multiprocessing_context=multiprocessing_method)
    valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, multiprocessing_context=multiprocessing_method)
    return train_loader, valid_loader


def get_logger(args, experiment_name):
    if args.disable_logger:
        return None
    else:
        if not args.project_name:
            raise AttributeError('Expected valid wandb project name. Logger can be disabled by --disable_logger.')
        os.environ["WANDB_API_KEY"] = 'b1968b98a10de3bf1f54b093a6398401d7f6741f'
        return pl.loggers.WandbLogger(project=args.project_name, save_code=False, name=experiment_name,
                                      group=args.group_name)


def load_pretrained_weights(args, model):
    print(f'LOADING PRETRAINED WEIGHTS FROM: {args.pretrained_weights}')
    # load whole model (maybe including effect).
    pretrained_model = model.__class__.load_from_checkpoint(args.pretrained_weights, strict=False)
    # "Steal" the prediction network with the pretrained weights.
    model.network = pretrained_model.network
    # discard the pretrained model
    pretrained_model.network = None
    del pretrained_model


def setup_and_start_training(args, model, base_dir):
    if args.pretrained_weights is not None:
        load_pretrained_weights(args, model)
    experiment_name, intermediate_results_folder = get_exp_name_and_interm_folder(args, base_dir)
    logger = get_logger(args, experiment_name)
    checkpoint_callback = ModelCheckpoint(dirpath=base_dir / 'checkpoints' / args.group_name / experiment_name,
                                          save_top_k=args.save_top_k,
                                          every_n_train_steps=args.save_model_each_n_training_steps,
                                          monitor="train/loss", verbose=True, save_last=True,
                                          save_on_train_epoch_end=True)
    devices = list(range(args.num_train_gpus))
    print(f'Using {devices} for training.')
    trainer = pl.Trainer(log_every_n_steps=50, accelerator='gpu', devices=devices,
                         max_epochs=args.epochs, logger=logger,
                         callbacks=[checkpoint_callback], gradient_clip_val=args.grad_clip,
                         strategy=get_distribution_strategy() if args.num_train_gpus > 1 else 'auto')
    train_loader, valid_loader = get_dataloaders(args.dataset_path, args.batch_size, args.num_workers,
                                                 model, args)

    trainer.fit(model, train_loader, valid_loader)


def get_distribution_strategy():
    try:
        from pytorch_lightning.strategies import DDPStrategy
        return DDPStrategy(find_unused_parameters=False, static_graph=True)
    except ImportError:
        pass
    try:
        from pytorch_lightning.plugins import DDPPlugin as DDPStrategy
        return DDPStrategy()
    except ImportError:
        pass
    print('Could not optimize distribution strategy!')
    return 'ddp'


def get_model_argparser():
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--style', help='style image',
                        default=f"{Path(__file__).parent}/../experiments/target/oilpaint_portrait.jpg")
    parser.add_argument('--img_size', help='dimension size of all images during training and validation', type=int,
                        default=256)
    parser.add_argument('--style_img_size', help='dimension size of the style image for the perceptual loss.', type=int,
                        default=256)
    # Model Params
    parser.add_argument('--architecture', help='Choose the PPN Model. '
                                               'The plain johnson network which contains bachnorm '
                                               'layers performed badly in the past.',
                        choices=[str(architecture) for architecture in list(PPNArchitectures)],
                        default='johnson_instance_norm')

    # Training Hyperparams
    parser.add_argument('--lr', help='Learning rate of the optimizer.', type=float,
                        default=1e-3)
    parser.add_argument('--use_vgg19_loss', help='Activates the VGG 19 loss network instead of the VGG16 loss network.',
                        action='store_true')
    parser.add_argument('--style_weight', help='Weight of the style loss.', type=float,
                        default=1e10)
    # Hint of relevance: https://github.com/gordicaleksa/pytorch-neural-style-transfer#impact-of-total-variation-tv-loss
    # What is TV: https://arxiv.org/pdf/1412.0035.pdf
    parser.add_argument('--tv_weight', type=float, default=0)

    # Checkpoints and results
    parser.add_argument('--save_model_each_n_training_steps', help='Save a model every n training steps. '
                                                                   '(sensitive to batch_size).', type=int,
                        default=2500)

    # Other params
    parser.add_argument('--num_train_gpus', help='Num GPUs to use. Default None will use all gpus available.'
                                                 ' 0 to disables GPU usage. Good for debugging.', type=int,
                        default=torch.cuda.device_count())
    parser.add_argument('--num_workers', help='Number of worker for dataloading for train and valid each.'
                                              'Setting it to 0 is good for debugging to avoid '
                                              'multiprocessing sideffects.', type=int,
                        default=os.cpu_count())

    return parser


def add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument('--experiment_name', help='Name of the training. Used for log identification. '
                                                  'Is normally inferred by all other parameters but can be set as well.',
                        type=str, default=None)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int,
                        default=8)
    parser.add_argument('--batch_size', help='Batch size during training and validation', type=int,
                        default=8)
    # Directories
    parser.add_argument('--dataset_path', help='Path to the dataset, which is split in train validation randomly.'
                                               'Should contain a large number of images. e.g. imagenet or ms coco',
                        type=str, default='/data/datasets/coco_stuff/images/train2017/')
    parser.add_argument('--logs_dir', help='The directory where to place the logs: logs and model checkpoints.',
                        type=str, default='../logs')

    parser.add_argument('--save_top_k', help='Save the top k results of the model based on the loss.', type=int,
                        default=1)

    parser.add_argument('--disable_logger', help='Disable Wandb logger.', action='store_true')
    parser.add_argument('--project_name', help='Wandb project name.', type=str, default='SSAST')
    parser.add_argument('--group_name', help='Wandb group name.', type=str, required=True)
    parser.add_argument('--grad_clip', help='Gradient Clipping value.', type=float, default=None)
    parser.add_argument('--pretrained_weights',
                        help='Path to the model checkpoint which contains the pretrained weights of the prediction network.',
                        type=str, default=None)

    return parser
