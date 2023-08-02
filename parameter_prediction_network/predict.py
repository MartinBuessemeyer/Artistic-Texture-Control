import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from local_ppn.data.image_folder import ImageFolder, default_loader
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel
from parameter_prediction_network.ppn_model import PPNModel

'''
The following script performs NST by loading a checkpoint of a trained PPN and its training configuration.
The training configuration ist part of the checkpoint. Currently only fixed image size predictions are supported.
This is because some image sizes result in inequal perdicted parameter map sizes.
Example call:
```
python -m parameter_prediction_network.predict --checkpoint logs/oilpaint/checkpoints/default_training_name/epoch=0-step=9.ckpt 
    --out_folder ./predictions 
    --input ./experiments/nprp/level1 --disable_gpu --num_workers 0
```
The command line arguments are explained in the argparse help.
'''


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None, return_paths=True):
        img = default_loader(img_path)
        dividable_factor = 8
        width, height = img.size
        width = (width // dividable_factor) * dividable_factor
        height = (height // dividable_factor) * dividable_factor
        print(f'Resizing from: ({img.size[0], img.size[1]}) to: ({width},{height})')
        img = img.resize((width, height))
        if transform is not None:
            self.img = transform(img)
        self.img_path = img_path
        self.return_paths = return_paths

    def __getitem__(self, index):
        if self.return_paths:
            return self.img, (str(self.img_path),)
        else:
            return self.img

    def __len__(self):
        return 1


def get_argparser():
    parser = argparse.ArgumentParser()
    # Predict related parameters
    parser.add_argument('--num_workers', help='Number of worker for dataloading for train and valid each.'
                                              'Setting it to 0 is good for debugging to avoid '
                                              'multiprocessing sideffects.', type=int, default=8)
    parser.add_argument('--disable_gpu', help='Disables GPU usage. Good for debugging.', action='store_true')
    parser.add_argument('--checkpoint', help='Path to the checkpoint file.', type=str, required=True)
    parser.add_argument('--out_folder', help='Path to the output folder where the resulting images are saved.')
    parser.add_argument('--input', help='Path to a folder containing images or path to a single image.')
    parser.add_argument('--johnson_nst', help='Uses Johnson NST instead of algorithmic style transfer.',
                        action='store_true')
    parser.add_argument('--use_scaled_masks', help='Uses scaled up masks by first resizing the image with '
                                                   'retained aspect ratio to 256 to predict the parameters and '
                                                   'then scale them to tho original resolution.',
                        action='store_true')
    return parser


def get_data_loader(datset_path: Path, num_workers: int, use_gpu: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if datset_path.is_dir():
        dataset = ImageFolder(root=datset_path, transform=transform, return_paths=True)
    else:
        dataset = SingleImageDataset(datset_path, transform=transform, return_paths=True)
    return DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=use_gpu)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    model_class = JohnsonNSTModel if args.johnson_nst else PPNModel
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.use_scaled_masks = args.use_scaled_masks
    data_loader = get_data_loader(Path(args.input), args.num_workers, args.disable_gpu)
    model.prediction_output_folder = Path(args.out_folder)
    model.prediction_output_folder.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(gpus=1 if not args.disable_gpu else 0)
    trainer.predict(model, data_loader)
