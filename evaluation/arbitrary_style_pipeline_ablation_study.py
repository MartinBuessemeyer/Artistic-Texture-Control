import itertools
import multiprocessing as mp
import pickle
import shutil
from copy import deepcopy, Error
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from art_fid.art_fid import compute_fid_infinity, compute_fid, compute_content_distance
from tqdm.auto import tqdm

from brushstroke.paint_transformer.inference.brushstroke_loss import BrushStrokeLoss
from effects.arbitrary_style_first_stage import perform_first_stage_segmentation
from evaluation.arbitrary_style_for_ablation_study import PipelineConfiguration, ArbitraryStyleAblationStudyEffect
from helpers import load_image_cuda, save_as_image, turn_on_deterministic_computation
from helpers.losses import TotalVariationLoss, PerceptualLoss
from helpers.visual_parameter_def import arbitrary_style_presets
from parameter_optimization.optimization_utils import ParameterContainer
from parameter_optimization.strotss_org import execute_style_transfer
from parameter_prediction_network.specific_losses.vgg16_loss import Vgg16Loss

# NON-CRITICAL SETTINGS (like folders, do not neccesarily change the results)

BASE_DIR = Path('./logs/ablation_study_watercolor_smoothed')
CONTENT_IMG_DIR = BASE_DIR / 'content_imgs'
STROTTS_RESULTS_DIR = BASE_DIR / 'strotts_results'
STROTTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_RESULT_DIR = BASE_DIR / 'optimization'
STYLE_IMG_DIR = BASE_DIR / 'popular_styles'

# CRITICAL SETTINGS (a change will lead to different results)

NUM_GPUS = torch.cuda.device_count()
NUM_GPUS = 1

# ablation experiments for optimization
pipeline_configurations: List[PipelineConfiguration] = [
    PipelineConfiguration('full_pipeline'),
    # PipelineConfiguration('full_pipeline_with_rc_loss', use_regional_consistency_losses=True),
    # PipelineConfiguration('no_bilateral', use_bilateral=False, use_regional_consistency_losses=True),
    # PipelineConfiguration('no_bump_mapping', use_bump_mapping=False, use_regional_consistency_losses=True),
    # PipelineConfiguration('no_xdog', use_xdog=False, use_regional_consistency_losses=True),
    # PipelineConfiguration('no_contrast', use_contrast=False, use_regional_consistency_losses=True)
]

# ablation study for ppn
# full pipeline with and without consistency loss
# result from study above with and without regional cosistency loss
RESUME = True
IMG_SIZE = 512
GATYS_LOSS_SETTINGS = {'style_img_size': 256, 'style_weight': 5e10}
# arbitrary style pipeline
AS_FIRST_STAGE_SETTINGS = {
    'kernel_size': 3, 'sigma': 1.0, 'n_segments': 5000
}

LEARNING_CONFIG = {'lr': 0.01, 'num_iters': 1000, 'brushstroke_weight': 0.01, 'tv_mask_weight': 0.2,
                   "lr_stop": 0.00005,
                   "lr_decay": 0.98,
                   "lr_decay_start": 50, "local_params": True, "sigma_large": 1.5,
                   "smoothing_steps": [10, 25, 50, 100, 250, 500], "img_size": None}


@dataclass
class NonArtFIDStudyMetrics:
    gatys_content_loss: float
    gatys_style_loss: float
    gatys_loss: float
    l1_loss_to_strotts: float
    brushstroke_loss: float
    tv_mask_loss: float


@dataclass
class StudyMetrics(NonArtFIDStudyMetrics):
    art_fid_score: float
    art_fid_fid_value: float
    art_fid_content_dist: float


def compute_as_pipeline_metrics(content_img: Path, style_img: Path, result_dir: Path,
                                result_mask_dir: Path,
                                pipeline_config: PipelineConfiguration, device):
    try:
        print(device)
        if (result_dir / content_img.name).is_file() and (result_dir / content_img.name).exists() and RESUME:
            return

        strotss_nst = load_image_cuda(get_strotts_path(style_img, content_img), long_edge=IMG_SIZE).to(device)
        effect = ArbitraryStyleAblationStudyEffect(pipeline_config).to(device)
        effect.disable_checkpoints()
        preset = deepcopy(arbitrary_style_presets)
        strotss_nst = strotss_nst.to(device)

        _, _, first_stage_img = perform_first_stage_segmentation(deepcopy(strotss_nst),
                                                                 AS_FIRST_STAGE_SETTINGS['kernel_size'],
                                                                 AS_FIRST_STAGE_SETTINGS['sigma'],
                                                                 AS_FIRST_STAGE_SETTINGS['n_segments'])
        content_image = load_image_cuda(content_img, long_edge=IMG_SIZE).to(device)
        strotts_size = list(strotss_nst.shape[-2:])
        content_image = torch.nn.functional.interpolate(content_image, strotts_size)

        first_stage_img = content_image.to(device)
        save_as_image(first_stage_img, result_mask_dir / f'first_stage_output_{content_img.name}.png', clamp=True)

        vp = effect.vpd.preset_tensor(preset, first_stage_img, True)
        grad_vp = ParameterContainer(vp, smooth=False)

        optimizer = torch.optim.Adam(list(grad_vp.parameters()), lr=LEARNING_CONFIG['lr'])

        l1_loss = torch.nn.L1Loss().to(device)
        brushstroke_loss = torch.jit.script(
            BrushStrokeLoss(brushstroke_weight=LEARNING_CONFIG['brushstroke_weight']).to(device))
        tv_loss = TotalVariationLoss(regularizer_weight=LEARNING_CONFIG['tv_mask_weight']).to(device)
        # Train
        print(f'Optimizing parameters of Image with size: {(first_stage_img.shape[2], first_stage_img.shape[3])}')
        lr = LEARNING_CONFIG["lr"]
        for i in tqdm(range(LEARNING_CONFIG['num_iters']), desc='Optimizing image parameters'):
            if i % 5 == 0 and i > LEARNING_CONFIG["lr_decay_start"]:
                lr = lr * LEARNING_CONFIG["lr_decay"]

                if lr < LEARNING_CONFIG["lr_stop"]:
                    lr = LEARNING_CONFIG["lr_stop"]

                # decay learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            stylized_img = effect(first_stage_img, grad_vp())
            loss = l1_loss(stylized_img, strotss_nst)
            if pipeline_config.use_regional_consistency_losses:
                visual_params = grad_vp() + 0.5
                loss += brushstroke_loss(visual_params)
                loss += tv_loss(visual_params)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save mask imgs
        params = grad_vp() + 0.5
        for mask_idx in range(params.shape[1]):
            mask_name = effect.vpd.vp_ranges[mask_idx][0]
            save_as_image(params[0, mask_idx], result_mask_dir / f'{content_img.name}_{mask_name}.png', clamp=True)

        # get final output
        with torch.no_grad():
            stylized_img = effect(first_stage_img, grad_vp())
            save_as_image(stylized_img, result_dir / content_img.name)

            gatys_loss = Vgg16Loss(style_img, GATYS_LOSS_SETTINGS['style_img_size'],
                                   GATYS_LOSS_SETTINGS['style_weight']).to(device)

            content_loss, style_loss = gatys_loss(load_image_cuda(content_img, long_edge=IMG_SIZE).to(device),
                                                  stylized_img)

            visual_params = grad_vp() + 0.5
    except Error as e:
        print(e)

    '''return NonArtFIDStudyMetrics(
        gatys_content_loss=content_loss.item(), gatys_style_loss=style_loss.item(),
        gatys_loss=(content_loss + style_loss).item(),
        l1_loss_to_strotts=l1_loss(stylized_img, strotss_nst).item(),
        brushstroke_loss=brushstroke_loss(visual_params).item(),
        tv_mask_loss=tv_loss(visual_params).item(),
    )'''


def calc_non_artfid_metrics(style_img, content_img, result_dir, result_mask_dir,
                            pipeline_config) -> NonArtFIDStudyMetrics:
    '''gatys_loss = Vgg16Loss(style_img, GATYS_LOSS_SETTINGS['style_img_size'],
                           GATYS_LOSS_SETTINGS['style_weight']).cuda()'''
    # perceptual_loss = PerceptualLoss(style_img, image_dim=1024, style_weight=1, content_weight=1).cuda()
    perceptual_loss = PerceptualLoss(style_img, image_dim=1024, style_weight=5e3, content_weight=1e-2).cuda()
    stylized_img = load_image_cuda(result_dir / content_img.name)
    strotss_nst = load_image_cuda(get_strotts_path(style_img, content_img), long_edge=IMG_SIZE).cuda()
    content_loss, style_loss = perceptual_loss(load_image_cuda(content_img, long_edge=IMG_SIZE), stylized_img)
    l1_loss = torch.nn.L1Loss().cuda()
    brushstroke_loss = BrushStrokeLoss(brushstroke_weight=LEARNING_CONFIG['brushstroke_weight']).cuda()
    tv_loss = TotalVariationLoss(regularizer_weight=LEARNING_CONFIG['tv_mask_weight']).cuda()
    effect = ArbitraryStyleAblationStudyEffect(pipeline_config).cuda()
    masks = []
    for mask_name, _, _ in effect.vpd.vp_ranges:
        masks.append(load_image_cuda(result_mask_dir / f'{content_img.name}_{mask_name}.png')[:, 0:1])
    visual_params = torch.cat(masks, dim=1)
    return NonArtFIDStudyMetrics(
        gatys_content_loss=content_loss.item(), gatys_style_loss=style_loss.item(),
        gatys_loss=(content_loss + style_loss).item(),
        l1_loss_to_strotts=l1_loss(stylized_img, strotss_nst).item(),
        brushstroke_loss=brushstroke_loss(visual_params).item(),
        tv_mask_loss=tv_loss(visual_params).item())


def get_strotts_path(style_img: Path, content_img: Path) -> Path:
    return STROTTS_RESULTS_DIR / f'{style_img.stem}_{content_img.stem}.png'


def perform_strotts_nst(style_img: Path, content_img: Path, device) -> torch.Tensor:
    strotss_out = get_strotts_path(style_img, content_img)
    if not strotss_out.is_file():
        print(device)
        strotss_nst = execute_style_transfer(content_img, style_img, 512, device=device)
        strotss_nst.save(strotss_out)
    strotss_nst = load_image_cuda(strotss_out, long_edge=512)
    save_as_image(strotss_nst, strotss_out, clamp=True)
    return strotss_nst


# COPIED AND ADJUSTED from the art_fid lib to return each component seperately
def compute_art_fid(path_to_stylized, path_to_style, path_to_content, batch_size, device, mode='art_fid_inf',
                    content_metric='lpips', num_workers=1):
    # print('Compute FID value...')
    if mode == 'art_fid_inf':
        fid_value = compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device=device,
                                         num_workers=num_workers)
    else:
        fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device=device, num_workers=num_workers)

    # print('Compute content distance...')
    content_dist = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric,
                                            device=device,
                                            num_workers=num_workers)

    art_fid_value = (content_dist + 1) * (fid_value + 1)
    return art_fid_value.item(), fid_value.item(), content_dist.item()


def perform_ablation_study():
    num_style_imgs = len(list(STYLE_IMG_DIR.iterdir()))
    aggregated_metrics = {}
    for configuration in tqdm(pipeline_configurations, desc='Pipeline Configurations'):
        aggregated_metrics[configuration.name] = {}
        metrics_list = []
        for style_img in tqdm(STYLE_IMG_DIR.iterdir(), desc='Style Images', total=num_style_imgs):
            experiment_base_dir = BASE_RESULT_DIR / configuration.name / style_img.stem
            result_dir = experiment_base_dir / 'output'
            result_dir.mkdir(parents=True, exist_ok=True)
            result_mask_dir = experiment_base_dir / 'param_masks'
            result_mask_dir.mkdir(parents=True, exist_ok=True)
            style_img_folder = experiment_base_dir / 'style_imgs'
            style_img_folder.mkdir(parents=True, exist_ok=True)

            ctx = mp.get_context('spawn')
            with ctx.Pool(NUM_GPUS * 1, maxtasksperchild=1) as pool:
                # construct iterator
                num_content_imgs = len(list(CONTENT_IMG_DIR.iterdir()))
                iterator = list(zip(CONTENT_IMG_DIR.iterdir(),
                                    itertools.repeat(style_img, num_content_imgs),
                                    itertools.repeat(result_dir, num_content_imgs),
                                    itertools.repeat(result_mask_dir, num_content_imgs),
                                    itertools.repeat(configuration, num_content_imgs),
                                    get_device_iterator(num_content_imgs)))

                pool.starmap(compute_as_pipeline_metrics, iterator)

            # compute metrics

            for content_img in CONTENT_IMG_DIR.iterdir():
                shutil.copy2(style_img, style_img_folder / f'{style_img.stem}_{content_img.name}')
            content_metrics_list = []
            for content_img in CONTENT_IMG_DIR.iterdir():
                content_metrics_list.append(
                    calc_non_artfid_metrics(style_img, content_img, result_dir, result_mask_dir, configuration))
            # art_fid
            ART_FID_SETTINGS = {'mode': 'art_fid_inf',
                                'content_metric': 'lpips'}

            art_fid_score, art_fid_fid_value, art_fid_content_dist = \
                compute_art_fid(path_to_style=str(style_img_folder), path_to_content=str(CONTENT_IMG_DIR),
                                path_to_stylized=str(result_dir),
                                batch_size=1, device='gpu', num_workers=0,
                                # batch_size=1, because of different image sizes
                                mode=ART_FID_SETTINGS['mode'], content_metric=ART_FID_SETTINGS['content_metric'])
            # compute avg metrics of content imgs
            metrics = StudyMetrics(
                float(np.mean([c_metric.gatys_content_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.gatys_style_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.gatys_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.l1_loss_to_strotts for c_metric in content_metrics_list])),
                float(np.mean([c_metric.brushstroke_loss for c_metric in content_metrics_list])),
                float(np.mean([c_metric.tv_mask_loss for c_metric in content_metrics_list])),
                art_fid_score, art_fid_fid_value, art_fid_content_dist
            )
            with open(experiment_base_dir / 'metrics.pickle', 'wb') as file:
                pickle.dump(metrics, file)
            metrics_list.append(metrics)
        # compute avg metrics and std of style imgs
        for metric_name in vars(metrics_list[0]).keys():
            metric_vals = [getattr(metrics, metric_name) for metrics in metrics_list]
            aggregated_metrics[configuration.name][f'{metric_name}_mean'] = np.mean(metric_vals)
            aggregated_metrics[configuration.name][f'{metric_name}_std'] = np.std(metric_vals)
        metrics_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
        metrics_df.to_csv(BASE_DIR / 'ablation_study_metrics.csv')


# compute avg metrics and std

def get_device_iterator(length):
    return [f'cuda:{i % NUM_GPUS}' for i in range(length)]


def generate_strotts_targets():
    for content_img in CONTENT_IMG_DIR.iterdir():
        img = load_image_cuda(content_img, long_edge=512)
        save_as_image(img, content_img, clamp=True)
        num_style_imgs = len(list(STYLE_IMG_DIR.iterdir()))

        ctx = mp.get_context('spawn')
        with ctx.Pool(NUM_GPUS) as pool:
            iterator = list(zip(STYLE_IMG_DIR.iterdir(),
                                itertools.repeat(content_img, num_style_imgs),
                                get_device_iterator(num_style_imgs)))
            pool.starmap(perform_strotts_nst, iterator)
        # for style_img in tqdm(STYLE_IMG_DIR.iterdir(), desc='Style Images', total=num_style_imgs):
        # perform_strotts_nst(style_img, content_img)


if __name__ == "__main__":
    turn_on_deterministic_computation()
    # generate_strotts_targets()
    perform_ablation_study()
