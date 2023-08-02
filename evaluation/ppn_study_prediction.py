import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from brushstroke.paint_transformer.inference.brushstroke_loss import BrushStrokeLoss
from effects import ArbitraryStyleEffect
from effects.arbitrary_style_first_stage import perform_first_stage_segmentation
from evaluation.arbitrary_style_pipeline_ablation_study import compute_art_fid
from helpers import load_image_cuda, save_as_image
from helpers.losses import TotalVariationLoss
from parameter_prediction_network.ast.ast_network import SANetForNST
from parameter_prediction_network.ast.ast_ppn_model import AST_PPN_MODEL
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel
from parameter_prediction_network.ppn_model import PPNModel
from parameter_prediction_network.specific_losses.vgg16_loss import Vgg16Loss

IMG_SIZE = 512
KERNEL_SIZE = 3
SIGMA = 1.0
NUM_SEGMENTS = 1000
GATYS_LOSS_SETTINGS = {'style_img_size': 256, 'style_weight': 5e10}
RC_LOSS_SETTINGS = {'brushstroke_weight': 0.01, 'tv_mask_weight': 0.2}
ART_FID_SETTINGS = {'mode': 'art_fid_inf',
                    'content_metric': 'lpips'}

BASE_DIR = Path('./logs/sanet_ablation_study_1000_segments')
CONTENT_IMG_DIR = BASE_DIR / 'content_imgs'
OUTPUT_DIR = BASE_DIR / 'prediction'
STYLE_IMG_DIR = BASE_DIR / 'popular_styles'
CONFIGURATION_NAME = 'full_pipeline'

STYLE_TO_PPN = {
    'candy': './logs/arbitrary_style/checkpoints/popular_styles_finetune/candy_arch_jin_ab_style_256/last.ckpt',
    'delaunay': './logs/arbitrary_style/checkpoints/popular_styles_finetune/delaunay_arch_jin_ab_style_sw_7.5e+10_256/last.ckpt',
    'Femme_nue_assise': './logs/arbitrary_style/checkpoints/popular_styles_finetune/Femme_nue_assise_arch_jin_ab_style_sw_1.0e+11_256/last-v1.ckpt',
    'mosaic': './logs/arbitrary_style/checkpoints/popular_styles_finetune/mosaic_arch_jin_ab_style_sw_2.5e+10_256/last.ckpt',
    'rain-princess-cropped': './logs/arbitrary_style/checkpoints/popular_styles_finetune/rain-princess-cropped_arch_jin_ab_style_sw_5.0e+10_256/last.ckpt',
    'starry_night': './logs/arbitrary_style/checkpoints/popular_styles_finetune/starry_night_arch_jin_ab_style_sw_5.0e+10_256/last.ckpt',
    'the_scream': './logs/arbitrary_style/checkpoints/popular_styles_finetune/the_scream_arch_jin_ab_style_sw_5.0e+10_256/last.ckpt',
    'the_shipwreck_of_the_minotaur': './logs/arbitrary_style/checkpoints/popular_styles_finetune/the_shipwreck_of_the_minotaur_arch_jin_ab_style_sw_1.0e+11_256/last.ckpt',
    'udnie': './logs/arbitrary_style/checkpoints/popular_styles_finetune/udnie_arch_jin_ab_style_sw_2.5e+10_256/last.ckpt',
    'wave': './logs/arbitrary_style/checkpoints/popular_styles_finetune/wave_arch_jin_ab_style_sw_5.0e+10_256/last.ckpt'}

STYLE_TO_JOHNSON_NST = {
    'candy': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/candy_arch_jin_johnson_nst_grad_clip_1.0e+06_256/last.ckpt',
    'delaunay': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/delaunay_arch_jin_johnson_nst_sw_7.5e+10_grad_clip_1.0e+06_256/last.ckpt',
    'Femme_nue_assise': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/Femme_nue_assise_arch_jin_johnson_nst_sw_1.0e+11_grad_clip_1.0e+06_256/last.ckpt',
    'mosaic': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/mosaic_arch_jin_johnson_nst_sw_2.5e+10_grad_clip_1.0e+06_256/last.ckpt',
    'rain-princess-cropped': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/rain-princess-cropped_arch_jin_johnson_nst_sw_5.0e+10_grad_clip_1.0e+06_256/last.ckpt',
    'starry_night': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/starry_night_arch_jin_johnson_nst_sw_5.0e+10_grad_clip_1.0e+06_256/last.ckpt',
    'the_scream': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/the_scream_arch_jin_johnson_nst_sw_5.0e+10_grad_clip_1.0e+06_256/last.ckpt',
    'the_shipwreck_of_the_minotaur': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/the_shipwreck_of_the_minotaur_arch_jin_johnson_nst_sw_1.0e+11_grad_clip_1.0e+06_256/last.ckpt',
    'udnie': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/udnie_arch_jin_johnson_nst_sw_2.5e+10_grad_clip_1.0e+06_256/last.ckpt',
    'wave': './logs/johnson_nst/checkpoints/popular_styles_adjusted_finetune/wave_arch_jin_johnson_nst_sw_5.0e+10_grad_clip_1.0e+06_256/last.ckpt'}

SANET_PPN_PATH = Path('./logs/SANet/checkpoints/train/sanet_lr_1e5/last.ckpt')


@dataclass
class NonArtFIDPPNStudyMetrics:
    gatys_content_loss: float
    gatys_style_loss: float
    gatys_loss: float
    johnson_gatys_content_loss: float
    johnson_gatys_style_loss: float
    johsnon_gatys_loss: float
    brushstroke_loss: float
    tv_mask_loss: float


@dataclass
class PPNStudyMetrics(NonArtFIDPPNStudyMetrics):
    art_fid_score: float
    art_fid_fid_value: float
    art_fid_content_dist: float


if __name__ == '__main__':
    with torch.no_grad():
        for content_img_path in CONTENT_IMG_DIR.iterdir():
            content_img = load_image_cuda(content_img_path, long_edge=512)
            orig_size = list(content_img.shape[-2:])
            # MUST BE DIVIDABLE BY 16
            DIVIDABLE = 16
            orig_size[0] = orig_size[0] - (orig_size[0] % DIVIDABLE)
            orig_size[1] = orig_size[1] - (orig_size[1] % DIVIDABLE)
            content_img = torch.nn.functional.interpolate(content_img, orig_size)
            save_as_image(content_img, content_img_path, clamp=True)

        metrics_list = []
        for style_img in tqdm(STYLE_IMG_DIR.iterdir(), total=len(list(STYLE_IMG_DIR.iterdir()))):
            experiment_base_dir = OUTPUT_DIR / style_img.stem
            style_img_folder = experiment_base_dir / 'style_imgs'
            result_dir = experiment_base_dir / 'output'
            mask_dir = experiment_base_dir / 'param_masks'
            style_img_folder.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            style_image = load_image_cuda(style_img)

            # single style ppn
            johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[style_img.stem]).cuda()
            ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[style_img.stem], strict=False).cuda()
            # arbitrary style SANet
            johnson_nst = SANetForNST().cuda()
            ppn: AST_PPN_MODEL = AST_PPN_MODEL.load_from_checkpoint(SANET_PPN_PATH).cuda()
            # TODO remove the dividable by 16 contraint line 86 ff.

            ppn.effect.disable_checkpoints()

            # perceptual_loss = PerceptualLoss(style_img, image_dim=1024, style_weight=5e3, content_weight=1e-2).cuda()
            perceptual_loss = Vgg16Loss(style_img, image_dim=256, style_weight=5e3, content_weight=1e-2).cuda()

            l1_loss = torch.nn.L1Loss().cuda()
            brushstroke_loss = BrushStrokeLoss(brushstroke_weight=RC_LOSS_SETTINGS['brushstroke_weight']).cuda()
            tv_loss = TotalVariationLoss(regularizer_weight=RC_LOSS_SETTINGS['tv_mask_weight']).cuda()

            assert ppn.effect.__class__ == ArbitraryStyleEffect

            content_metrics_list = []
            for content_img_path in tqdm(CONTENT_IMG_DIR.iterdir(), total=len(list(CONTENT_IMG_DIR.iterdir()))):
                content_img = load_image_cuda(content_img_path)
                orig_size = list(content_img.shape[-2:])

                nst_img = johnson_nst(content_img, style_image)
                # nst_img = johnson_nst(content_img)
                nst_img = torch.nn.functional.interpolate(nst_img, orig_size)
                nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

                nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, KERNEL_SIZE, SIGMA,
                                                                               NUM_SEGMENTS)

                parameters, _ = ppn.predict_params(segmented_img, style_image)
                # parameters = ppn.predict_params(segmented_img)
                parameters = torch.nn.functional.interpolate(parameters, orig_size)
                parameters = torch.clamp(parameters, min=-0.5, max=0.5)
                stylized_image = ppn.effect(segmented_img, parameters)

                save_as_image(stylized_image, result_dir / content_img_path.name)

                params = parameters + 0.5
                for mask_idx in range(params.shape[1]):
                    mask_name = ppn.effect.vpd.vp_ranges[mask_idx][0]
                    save_as_image(params[0, mask_idx],
                                  mask_dir / f'{content_img_path.name}_{mask_name}.png',
                                  clamp=True)

                content_loss, style_loss = perceptual_loss(content_img,
                                                           stylized_image)
                johnson_content_loss, johnson_style_loss = perceptual_loss(content_img,
                                                                           nst_image)
                content_metrics_list.append(NonArtFIDPPNStudyMetrics(
                    gatys_content_loss=content_loss.item(),
                    gatys_style_loss=style_loss.item(),
                    gatys_loss=(content_loss + style_loss).item(),
                    johnson_gatys_content_loss=johnson_content_loss.item(),
                    johnson_gatys_style_loss=johnson_style_loss.item(),
                    johsnon_gatys_loss=(johnson_content_loss + johnson_style_loss).item(),
                    brushstroke_loss=brushstroke_loss(params).item(),
                    tv_mask_loss=tv_loss(params).item()))

            # compute artFID
            for content_img_path in CONTENT_IMG_DIR.iterdir():
                save_as_image(style_image, style_img_folder / content_img_path.name)

            art_fid_score, art_fid_fid_value, art_fid_content_dist = \
                compute_art_fid(path_to_style=str(style_img_folder), path_to_content=str(CONTENT_IMG_DIR),
                                path_to_stylized=str(result_dir),
                                batch_size=1, device='gpu', num_workers=0,
                                mode=ART_FID_SETTINGS['mode'], content_metric=ART_FID_SETTINGS['content_metric'])

            metrics = PPNStudyMetrics(
                float(np.mean([c_metric.gatys_content_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.gatys_style_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.gatys_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.johnson_gatys_content_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.johnson_gatys_style_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.johsnon_gatys_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.brushstroke_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.tv_mask_loss for c_metric in content_metrics_list])),
                art_fid_score, art_fid_fid_value, art_fid_content_dist
            )
            with open(experiment_base_dir / 'metrics.pickle', 'wb') as file:
                pickle.dump(metrics, file)
            metrics_list.append(metrics)
        # compute avg metrics and std of style imgs
        aggregated_metrics = {CONFIGURATION_NAME: {}}
        for metric_name in vars(metrics_list[0]).keys():
            metric_vals = [getattr(metrics, metric_name) for metrics in metrics_list]
            aggregated_metrics[CONFIGURATION_NAME][f'{metric_name}_mean'] = np.mean(metric_vals)
            aggregated_metrics[CONFIGURATION_NAME][f'{metric_name}_std'] = np.std(metric_vals)
        metrics_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
        metrics_df.to_csv(BASE_DIR / 'ablation_study_metrics.csv')
