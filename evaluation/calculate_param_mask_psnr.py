from pathlib import Path

import numpy as np
from skimage.restoration import estimate_sigma
from tqdm.auto import tqdm

from brushstroke.paint_transformer.inference.brushstroke_loss import BrushStrokeLoss
from helpers import load_image_cuda, torch_to_np
from helpers.losses import TotalVariationLoss

BASE_DIR = Path(
    './logs/sanet_ablation_study_5000_segments/prediction')
LEARNING_CONFIG = {'lr': 0.01, 'num_iters': 500, 'brushstroke_weight': 0.01, 'tv_mask_weight': 0.2}

if __name__ == '__main__':
    print(BASE_DIR)
    brushstroke_loss = BrushStrokeLoss(brushstroke_weight=LEARNING_CONFIG['brushstroke_weight']).cuda()
    tv_loss = TotalVariationLoss(regularizer_weight=LEARNING_CONFIG['tv_mask_weight']).cuda()

    noise_ratios = []
    brushstroke_losses = []
    tv_losses = []
    for pipeline_config_dir in tqdm(list(BASE_DIR.iterdir())):
        for param_mask_path in tqdm(list((pipeline_config_dir / 'param_masks').iterdir())):
            if 'shininess' in param_mask_path.name:
                continue
            param_mask_torch = load_image_cuda(param_mask_path)[:, 0:1, ...]
            param_mask = torch_to_np(param_mask_torch)
            assert len(list(param_mask.shape)) == 2
            noise_ratios.append(estimate_sigma(param_mask))
            tv_losses.append(tv_loss(param_mask_torch).item())
            brushstroke_losses.append(brushstroke_loss(param_mask_torch).item())
    print(f'noise: {np.mean(noise_ratios)}, tv mask: {np.mean(tv_losses)}, brushstroke: {np.mean(brushstroke_losses)}')
