import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, List

import torch
from torchvision.transforms import Resize, transforms

from brushstroke.paint_transformer.inference.brushstroke_loss import BrushStrokeLoss
from effects import get_default_settings, ArbitraryStyleEffect
from helpers import save_as_image, color_conversion
from helpers.losses import TotalVariationLoss
from parameter_prediction_network.arbitrary_style_helpers import JohnsonNSTTransform, \
    get_dataloader_device
from parameter_prediction_network.model_base import ModelBase


def create_hue_image(hue: torch.Tensor) -> torch.Tensor:
    tensor_shape = [1, 3] + list(hue.shape)
    img_tensor = torch.ones(tensor_shape)
    img_tensor[0:1] = hue
    return color_conversion.hsv_to_rgb(img_tensor)[0]


class PPNModel(ModelBase):
    def __init__(self, args: argparse.Namespace, base_dir: Path):
        self.save_hyperparameters()
        effect, _, _ = get_default_settings(args.effect)
        super().__init__(args, base_dir, len(effect.vpd.vp_ranges), (-0.5, 0.5))
        if not args.disable_gradient_checkpoints:
            effect.enable_checkpoints()
        self.effect = effect
        self.use_scaled_masks = False

        # Additinal losses
        self.use_l1_loss = args.use_l1_loss
        self.l1_loss = torch.nn.L1Loss()
        self.brushstroke_loss = BrushStrokeLoss(brushstroke_weight=args.brushstroke_weight)
        if not args.disable_jit_brushstroke:
            self.brushstroke_loss = torch.jit.script(self.brushstroke_loss)
        self.mask_tv_loss = TotalVariationLoss(regularizer_weight=args.mask_tv_weight)

    def predict_params(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = self.predict_params(x)
        return self.effect(x, parameters)

    def compute_separate_loss_terms(self, input_image: torch.Tensor, content_image: torch.Tensor,
                                    nst_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        separate_losses = {}
        params = self.predict_params(input_image)
        scaled_params = params + 0.5
        y_hat = self.effect(input_image, params)
        if self.use_l1_loss:
            separate_losses['l1_loss'] = self.l1_loss(nst_image, y_hat)
        else:
            separate_losses['content_loss'], separate_losses['style_loss'] = self.perceptual_loss(content_image, y_hat)
        separate_losses['regularizer_loss'] = self.tv_loss(y_hat)
        separate_losses['brushstroke_loss'] = self.brushstroke_loss(scaled_params)
        separate_losses['mask_tv_loss'] = self.mask_tv_loss(scaled_params)
        return separate_losses

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        img, ((img_path,),) = batch
        if self.use_scaled_masks:
            width, height = img.shape[2:]
            downscale = Resize(256)
            upscale = Resize((width, height))
            params = self.predict_params(downscale(img))
            params = upscale(params)
        else:
            params = self.predict_params(img)
        result = self.effect(img, params)
        save_as_image(result, self.prediction_output_folder / Path(img_path).name, clamp=True)

    def save_intermediate_train_result(self, x, debug=False):
        super().save_intermediate_train_result(x, debug)
        params_folder = self.intermediate_results_folder / 'param_masks'
        params_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile((self.intermediate_results_folder /
                         f'step_{self.train_idx}_batch_idx_0_.png').resolve(),
                        (params_folder / f'step_{self.train_idx}_result.png').resolve())
        statistics = ['STATS:\n\n']
        with torch.no_grad():
            params = self.predict_params(x[0:1])[0]
            for param_idx in range(params.size(0)):
                param = params[param_idx] + 0.5
                param_name = self.effect.vpd.vp_ranges[param_idx][0]
                statistics.append(
                    f'{param_name}:{" " * (25 - len(param_name))} mean: {param.mean()}, std:{param.std()}\n')
                save_as_image(param if 'hue' not in param_name.lower() else create_hue_image(param),
                              params_folder / f'step_{self.train_idx}_{param_name}.png', clamp=True)
        with open(params_folder / f'stats_step_{self.train_idx}.txt', 'wt') as stat_file:
            stat_file.writelines(statistics)

    def get_dataloader_transforms(self, args: argparse.Namespace) -> List[torch.nn.Module]:
        transform_list = [transforms.Resize(args.img_size),
                          transforms.CenterCrop(args.img_size),
                          transforms.ToTensor()]
        if self.effect.__class__ == ArbitraryStyleEffect:
            transform_list.append(JohnsonNSTTransform(args.johnson_nst_model, get_dataloader_device(args)))
        return transform_list
