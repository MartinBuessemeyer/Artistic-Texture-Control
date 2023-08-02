import argparse
from pathlib import Path
from typing import List, Any, Optional, Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.transforms import transforms

from helpers import save_as_image
from helpers.losses import PerceptualLoss, TotalVariationLoss
from parameter_prediction_network.path_utils import get_exp_name_and_interm_folder
from parameter_prediction_network.ppn_architectures import PPNArchitectures
from parameter_prediction_network.ppn_architectures.ppn_architectures import get_ppn_architecture
from parameter_prediction_network.specific_losses.vgg16_loss import Vgg16Loss


class ModelBase(pl.LightningModule):
    NUM_INTERMEDIATE_IMGS = 5

    def __init__(self, args: argparse.Namespace, base_dir: Path, num_outs: int,
                 normalization_range: Tuple[float, float]):
        super().__init__()
        _, intermediate_results_folder = get_exp_name_and_interm_folder(args, base_dir)
        self.save_hyperparameters()
        self.network = get_ppn_architecture(num_outs, PPNArchitectures[args.architecture])

        self.perceptual_loss = PerceptualLoss(args.style, args.style_img_size,
                                              style_weight=args.style_weight) if args.use_vgg19_loss else \
            Vgg16Loss(args.style, args.style_img_size, style_weight=args.style_weight)
        self.perceptual_loss.eval()
        self.tv_loss = TotalVariationLoss(args.tv_weight)
        self.lr = args.lr
        self.intermediate_results_folder = intermediate_results_folder

        self.prediction_output_folder = getattr(args, 'prediction_output_folder', None)

        self.save_img_every_n_train_steps = args.save_model_each_n_training_steps

        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), persistent=False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), persistent=False)
        self.train_idx = 0
        assert normalization_range[0] < normalization_range[1]
        self.normalization_range = normalization_range

    # Extracts the input image and content image from the batch.
    # Needed because of different behavior between arbitrary style pipline and other pipelines.
    def split_batch_to_images(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = batch
        if isinstance(x, list):
            input_batch, content_batch, nst_batch = x
        else:
            input_batch, content_batch, nst_batch = x, x, x
        return input_batch, content_batch, nst_batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        out = self.network(x)

        out = torch.sigmoid(out)
        out = out * (self.normalization_range[1] - self.normalization_range[0])
        out = out + self.normalization_range[0]
        return out

    def compute_separate_loss_terms(self, input_image: torch.Tensor, content_image: torch.Tensor,
                                    nst_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        separate_losses = {}
        y_hat = self(input_image)
        separate_losses['content_loss'], separate_losses['style_loss'] = self.perceptual_loss(content_image, y_hat)
        separate_losses['regularizer_loss'] = self.tv_loss(y_hat)
        return separate_losses

    def compute_loss(self, input_image: torch.Tensor, content_image: torch.Tensor, nst_image: torch.Tensor,
                     step_name: str) -> torch.Tensor:
        separate_losses = self.compute_separate_loss_terms(input_image, content_image, nst_image)
        combined_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss in separate_losses.items():
            self.log(f'{step_name}/{loss_name}', loss, logger=True, on_step=True)
            combined_loss += loss
        self.log(f'{step_name}/loss', combined_loss, logger=True, on_step=True)
        return combined_loss

    def training_step(self, batch: List[torch.Tensor], *args: List[Any]) -> STEP_OUTPUT:
        input_batch, content_batch, nst_batch = self.split_batch_to_images(batch)
        loss = self.compute_loss(input_batch, content_batch, nst_batch, 'train')
        self.train_idx += 1
        if self.train_idx % self.save_img_every_n_train_steps == 0:
            self.save_intermediate_train_result(input_batch[:ModelBase.NUM_INTERMEDIATE_IMGS],
                                                content_batch[:ModelBase.NUM_INTERMEDIATE_IMGS])
        # debug loss spikes.
        '''if self.train_idx > 100 and loss > 1e7:
            print(f'LOSS SPIKE DETECTED AT STEP: {self.train_idx}')
            self.save_intermediate_train_result(x, debug=True)'''
        return loss

    def validation_step(self, batch: List[torch.Tensor], *args: List[Any]) -> Optional[STEP_OUTPUT]:
        input_batch, content_batch, nst_batch = self.split_batch_to_images(batch)
        loss = self.compute_loss(input_batch, content_batch, nst_batch, 'valid')
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        img, ((img_path,),) = batch
        res = self(img)
        save_as_image(res, self.prediction_output_folder / Path(img_path).name, clamp=True)

    def predict_keep_img_size(self, x: torch.Tensor):
        y = self(x)
        return torch.nn.functional.interpolate(y, (x.size(2), x.size(3)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return [optimizer], []

    def get_dataloader_transforms(self, args: argparse.Namespace) -> List[torch.nn.Module]:
        return [transforms.Resize(args.img_size),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor()]

    def save_intermediate_train_result(self, x, content, debug=False):
        print(f'\nSaving intermediate result images from training to {self.intermediate_results_folder}.')
        with torch.no_grad():
            y = self(x)
            for batch_idx in range(y.shape[0]):
                combined_shape = list(y.shape[1:])
                y_width = combined_shape[2]
                combined_shape[2] = 3 * y_width
                combined_img = torch.empty(combined_shape, dtype=x.dtype, device=x.device)
                combined_img[:, :, y_width * 0:y_width * 1] = content[batch_idx]
                combined_img[:, :, y_width * 1:y_width * 2] = x[batch_idx]
                combined_img[:, :, y_width * 2:y_width * 3] = y[batch_idx]
                save_as_image(combined_img, self.intermediate_results_folder /
                                            f'step_{self.train_idx}_batch_idx_{batch_idx}_.png' if not debug else
                f'loss_spike_error_step_{self.train_idx}_batch_idx_{batch_idx}_.png', clamp=True)
