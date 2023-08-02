# TAKEN AND ADAPTED FROM: https://github.com/GlebSBrykin/SANET/blob/master/Train.ipynb

import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.transforms import Resize, transforms

from brushstroke.paint_transformer.inference.brushstroke_loss import BrushStrokeLoss
from effects import get_default_settings
from helpers import save_as_image
from helpers.losses import TotalVariationLoss
from parameter_prediction_network.arbitrary_style_helpers import SANetNSTTransform, get_dataloader_device
from parameter_prediction_network.ast.ast_network import get_vgg, \
    Transform, get_decoder, calc_mean_std, mean_variance_norm
from parameter_prediction_network.path_utils import get_exp_name_and_interm_folder


class AST_PPN_MODEL(pl.LightningModule):
    NUM_INTERMEDIATE_IMGS = 5

    def __init__(self, args: argparse.Namespace, base_dir: Path):
        self.save_hyperparameters()
        super(AST_PPN_MODEL, self).__init__()
        _, intermediate_results_folder = get_exp_name_and_interm_folder(args, base_dir)
        self.lr = args.lr
        self.intermediate_results_folder = intermediate_results_folder
        self.save_img_every_n_train_steps = args.save_model_each_n_training_steps
        assert args.effect == 'arbitrary_style'
        effect, _, _ = get_default_settings(args.effect)
        if not args.disable_gradient_checkpoints:
            effect.enable_checkpoints()
        self.effect = effect

        # SANet
        num_outs = len(effect.vpd.vp_ranges)
        # encoder
        encoder = get_vgg()
        encoder.load_state_dict(torch.load('./parameter_prediction_network/ast/weights/vgg_normalised.pth'))
        encoder = torch.nn.Sequential(*list(encoder.children())[:44])
        enc_layers = list(encoder.children())
        self.enc_1 = torch.nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = torch.nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = torch.nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = torch.nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = torch.nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # transform
        self.transform = Transform(in_planes=512)
        # decoder
        self.decoder = get_decoder(num_outs)
        # Losses
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight
        self.mse_loss = torch.nn.MSELoss()

        # Additinal losses
        self.brushstroke_loss = BrushStrokeLoss(brushstroke_weight=args.brushstroke_weight)
        if not args.disable_jit_brushstroke:
            self.brushstroke_loss = torch.jit.script(self.brushstroke_loss)
        self.mask_tv_loss = TotalVariationLoss(regularizer_weight=args.mask_tv_weight)
        self.tv_loss = TotalVariationLoss(args.tv_weight)
        self.save_img_every_n_train_steps = args.save_model_each_n_training_steps
        self.train_idx = 0

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input: torch.Tensor):
        encoded_1 = self.enc_1(input)
        encoded_2 = self.enc_2(encoded_1)
        encoded_3 = self.enc_3(encoded_2)
        encoded_4 = self.enc_4(encoded_3)
        encoded_5 = self.enc_5(encoded_4)
        return [encoded_1, encoded_2, encoded_3, encoded_4, encoded_5]

    def predict_params(self, first_stage_out: torch.Tensor, styles: torch.Tensor) -> Tuple[
        torch.Tensor, List[torch.Tensor]]:
        style_feats = self.encode_with_intermediate(styles)
        first_stage_feats = self.encode_with_intermediate(first_stage_out)
        stylized = self.transform(first_stage_feats[3], style_feats[3], first_stage_feats[4], style_feats[4])
        params = self.decoder(stylized)
        params = torch.sigmoid(params) - 0.5
        return params, style_feats

    def forward(self, first_stage_out: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        parameters, _ = self.predict_params(first_stage_out, styles)
        return self.effect(first_stage_out, parameters)

    def training_step(self, batch: List[torch.Tensor], *args: List[Any]) -> STEP_OUTPUT:
        in_batch, _ = batch
        first_stage_batch, content_batch, style_batch = in_batch
        loss = self.compute_loss(first_stage_batch, content_batch, style_batch, 'train')
        self.train_idx += 1
        if self.train_idx % self.save_img_every_n_train_steps == 0:
            self.save_intermediate_train_result(first_stage_batch[:AST_PPN_MODEL.NUM_INTERMEDIATE_IMGS],
                                                content_batch[:AST_PPN_MODEL.NUM_INTERMEDIATE_IMGS],
                                                style_batch[:AST_PPN_MODEL.NUM_INTERMEDIATE_IMGS])
        return loss

    def validation_step(self, batch: List[torch.Tensor], *args: List[Any]) -> Optional[STEP_OUTPUT]:
        in_batch, _ = batch
        first_stage_batch, content_batch, style_batch = in_batch
        loss = self.compute_loss(first_stage_batch, content_batch, style_batch, 'valid')
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError('TODO')
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.decoder.parameters()},
            {'params': self.transform.parameters()}], lr=self.lr)
        return [optimizer], []

    def compute_separate_loss_terms(self, first_stage_out: torch.Tensor, contents: torch.Tensor,
                                    styles: torch.Tensor) -> Dict[str, torch.Tensor]:
        separate_losses = {}
        params, style_feats = self.predict_params(first_stage_out, styles)
        scaled_params = params + 0.5
        y_hat = self.effect(first_stage_out, params)

        separate_losses['content_loss'], separate_losses['style_loss'], separate_losses[
            'identity_loss'] = self.compute_sanet_loss(y_hat, styles, contents, style_feats, first_stage_out)

        separate_losses['regularizer_loss'] = self.tv_loss(y_hat)
        separate_losses['brushstroke_loss'] = self.brushstroke_loss(scaled_params)
        separate_losses['mask_tv_loss'] = self.mask_tv_loss(scaled_params)
        return separate_losses

    def compute_loss(self, first_stage_out: torch.Tensor, contents: torch.Tensor, styles: torch.Tensor,
                     step_name: str) -> torch.Tensor:
        separate_losses = self.compute_separate_loss_terms(first_stage_out, contents, styles)
        combined_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss in separate_losses.items():
            self.log(f'{step_name}/{loss_name}', loss, logger=True, on_step=True)
            combined_loss += loss
        self.log(f'{step_name}/loss', combined_loss, logger=True, on_step=True)
        return combined_loss

    def compute_sanet_loss(self, y_hat: torch.Tensor, styles: torch.Tensor, contents: torch.Tensor,
                           style_feats: torch.Tensor, first_stage_out: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        content_feats = self.encode_with_intermediate(contents)
        g_t_feats = self.encode_with_intermediate(y_hat)

        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        """IDENTITY LOSSES"""
        """content_reconstruction_params = self.decoder(
            self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        content_reconstruction_params = torch.sigmoid(content_reconstruction_params) - 0.5
        Icc = self.effect(first_stage_out, content_reconstruction_params)
        style_reconstruction_params = self.decoder(
            self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        style_reconstruction_params = torch.sigmoid(style_reconstruction_params) - 0.5
        Iss = self.effect(first_stage_out, style_reconstruction_params)

        l_identity1 = self.calc_content_loss(Icc, contents) + self.calc_content_loss(Iss, styles)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])"""

        loss_c = self.content_weight * loss_c
        loss_s = self.style_weight * loss_s
        # loss_identity = l_identity1 * 50 + l_identity2 * 1
        return loss_c, loss_s, 0.0

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def save_intermediate_train_result(self, first_stage_out, content, styles):
        # save stylized imgs
        with torch.no_grad():
            print(f'\nSaving intermediate result images from training to {self.intermediate_results_folder}.')
            y = self(first_stage_out, styles)
            for batch_idx in range(y.shape[0]):
                combined_shape = list(y.shape[1:])
                y_width = combined_shape[2]
                combined_shape[2] = 4 * y_width
                combined_img = torch.empty(combined_shape, dtype=first_stage_out.dtype, device=first_stage_out.device)
                combined_img[:, :, y_width * 0:y_width * 1] = content[batch_idx]
                combined_img[:, :, y_width * 1:y_width * 2] = styles[batch_idx]
                combined_img[:, :, y_width * 2:y_width * 3] = first_stage_out[batch_idx]
                combined_img[:, :, y_width * 3:y_width * 4] = y[batch_idx]
                save_as_image(combined_img, self.intermediate_results_folder /
                              f'step_{self.train_idx}_batch_idx_{batch_idx}_.png', clamp=True)
        # save param masks and stats
        params_folder = self.intermediate_results_folder / 'param_masks'
        params_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile((self.intermediate_results_folder /
                         f'step_{self.train_idx}_batch_idx_0_.png').resolve(),
                        (params_folder / f'step_{self.train_idx}_result.png').resolve())
        statistics = ['STATS:\n\n']
        with torch.no_grad():
            params = self.predict_params(first_stage_out[0:1], styles[0:1])[0][0]
            for param_idx in range(params.size(0)):
                param = params[param_idx] + 0.5
                param_name = self.effect.vpd.vp_ranges[param_idx][0]
                statistics.append(
                    f'{param_name}:{" " * (25 - len(param_name))} mean: {param.mean()}, std:{param.std()}\n')
                save_as_image(param, params_folder / f'step_{self.train_idx}_{param_name}.png', clamp=True)
        with open(params_folder / f'stats_step_{self.train_idx}.txt', 'wt') as stat_file:
            stat_file.writelines(statistics)

    def get_dataloader_transforms(self, args: argparse.Namespace) -> List[torch.nn.Module]:
        return [
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            SANetNSTTransform(args.wikiart_dir, get_dataloader_device(args))
        ]
