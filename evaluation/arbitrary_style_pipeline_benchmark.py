import time
from copy import deepcopy
from pathlib import Path

import torch
from torchvision import transforms
from tqdm.auto import tqdm

from effects import ArbitraryStyleEffect
from effects.arbitrary_style_first_stage import perform_first_stage_segmentation
from evaluation.arbitrary_style_pipeline_ablation_study import get_strotts_path
from helpers import load_image_cuda
from helpers.visual_parameter_def import arbitrary_style_presets
from parameter_optimization.optimization_utils import ParameterContainer
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel
from parameter_prediction_network.ppn_model import PPNModel

# NON-CRITICAL SETTINGS (like folders, do not neccesarily change the results)

BASE_DIR = Path('./logs/ablation_study_5000_segments_correct_gatys_loss')
CONTENT_IMG_DIR = BASE_DIR / 'content_imgs'
STROTTS_RESULTS_DIR = BASE_DIR / 'strotts_results'
STROTTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_RESULT_DIR = BASE_DIR / 'optimization'
STYLE_IMG_DIR = BASE_DIR / 'popular_styles'

# CRITICAL SETTINGS (a change will lead to different results)

# ablation study for ppn
# full pipeline with and without consistency loss
# result from study above with and without regional cosistency loss
KERNEL_SIZE = 3
SIGMA = 1.0
NUM_SEGMENTS = 5000
AS_FIRST_STAGE_SETTINGS = {
    'kernel_size': 3, 'sigma': 1.0, 'n_segments': 5000
}
LEARNING_CONFIG = {'lr': 0.01, 'num_iters': 5, 'brushstroke_weight': 0.01, 'tv_mask_weight': 0.2}

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


def optimization_benchmark(img_size):
    content_img_path = Path(
        './logs/ablation_study_5000_segments_correct_gatys_loss/content_imgs/000000096463.jpg')
    device = 'cuda'
    STYLE_IMGS = list(STYLE_IMG_DIR.iterdir())[:1]
    time_spent_optimizing = 0
    for style_img in tqdm(STYLE_IMGS, desc='Style Images'):
        strotts_nst = load_image_cuda(get_strotts_path(style_img, content_img_path))
        content_img = load_image_cuda(content_img_path)
        strotts_nst = torch.nn.functional.interpolate(strotts_nst, (img_size, img_size))
        content_img = torch.nn.functional.interpolate(content_img, (img_size, img_size))
        strotts_tmp = deepcopy(strotts_nst)

        torch.cuda.reset_peak_memory_stats()
        start = time.time_ns()
        _, _, first_stage_img = perform_first_stage_segmentation(strotts_tmp, AS_FIRST_STAGE_SETTINGS['kernel_size'],
                                                                 AS_FIRST_STAGE_SETTINGS['sigma'],
                                                                 AS_FIRST_STAGE_SETTINGS['n_segments'])
        time_spent_optimizing += time.time_ns() - start
        print("torch.cuda.max_memory_allocated: %fGB" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

        effect = ArbitraryStyleEffect().to(device)
        preset = deepcopy(arbitrary_style_presets)
        content_img = first_stage_img

        vp = effect.vpd.preset_tensor(preset, content_img, True)
        grad_vp = ParameterContainer(vp, smooth=False)

        optimizer = torch.optim.Adam(list(grad_vp.parameters()), lr=LEARNING_CONFIG['lr'])

        l1_loss = torch.nn.L1Loss().to(device)
        '''brushstroke_loss = torch.jit.script(
            BrushStrokeLoss(brushstroke_weight=LEARNING_CONFIG['brushstroke_weight']).to(device))
        tv_loss = TotalVariationLoss(regularizer_weight=LEARNING_CONFIG['tv_mask_weight']).to(device)'''
        torch.cuda.reset_peak_memory_stats()
        effect.disable_checkpoints()
        start = time.time_ns()
        for _ in range(LEARNING_CONFIG['num_iters']):
            stylized_img = effect(content_img, grad_vp())
            loss = l1_loss(stylized_img, strotts_nst)
            '''visual_params = grad_vp() + 0.5
            loss += brushstroke_loss(visual_params)
            loss += tv_loss(visual_params)'''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_spent_optimizing += time.time_ns() - start
        print("torch.cuda.max_memory_allocated: %fGB" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

    time_in_seconds = time_spent_optimizing / (10 ** 9)
    return time_in_seconds / len(STYLE_IMGS)


def ppn_benchmark(img_size):
    time_spent_predicting = 0
    STYLE_IMGS = list(STYLE_IMG_DIR.iterdir())[:1]
    with torch.no_grad():
        for style_img in tqdm(STYLE_IMGS, desc='Style Images'):
            johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[style_img.stem]).cuda()
            ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[style_img.stem], strict=False).cuda()
            ppn.effect.disable_checkpoints()
            '''sanet_nst = SANetForNST().cuda()
            ast_ppn: AST_PPN_MODEL = AST_PPN_MODEL.load_from_checkpoint(SANET_PPN_PATH).cuda()'''
            style_img = load_image_cuda(style_img, 512)
            style_transforms = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.RandomCrop(256),
            ])
            style_image = style_transforms(style_img)
            # warm up
            for content_img_path in tqdm(list(CONTENT_IMG_DIR.iterdir()),
                                         total=len(list(CONTENT_IMG_DIR.iterdir()))):
                content_img = load_image_cuda(content_img_path)
                content_img = torch.nn.functional.interpolate(content_img, (img_size, img_size))
                # start = time.time_ns()
                nst_img = johnson_nst(content_img)
                # nst_img = sanet_nst(content_img, style_image)
                nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

                _, _, segmented_img = perform_first_stage_segmentation(nst_img, KERNEL_SIZE, SIGMA, NUM_SEGMENTS)
                # stylized_image = ast_ppn(segmented_img, style_image)
                # stylized_image = ppn(segmented_img)
            torch.cuda.reset_peak_memory_stats()
            ppn.effect.disable_checkpoints()
            for content_img_path in tqdm(CONTENT_IMG_DIR.iterdir(), total=len(list(CONTENT_IMG_DIR.iterdir()))):
                content_img = load_image_cuda(content_img_path)
                content_img = torch.nn.functional.interpolate(content_img, (img_size, img_size))

                # start = time.time_ns()

                nst_img = johnson_nst(content_img)
                # nst_img = sanet_nst(content_img, style_image)
                nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

                _, _, segmented_img = perform_first_stage_segmentation(nst_img, KERNEL_SIZE, SIGMA, NUM_SEGMENTS)
                start = time.time_ns()
                # stylized_image = ast_ppn(segmented_img, style_image)
                stylized_image = ppn(segmented_img)
                time_spent_predicting += time.time_ns() - start
            print("torch.cuda.max_memory_allocated: %fGB" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

        time_in_seconds = time_spent_predicting / (10 ** 9)
        return time_in_seconds / (len(STYLE_IMGS) * len(list(CONTENT_IMG_DIR.iterdir())))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    # profile_ppn(512)
    IMAGE_SIZES = [256, 512, 1024]
    for image_size in IMAGE_SIZES:
        # print(f'OPTMIZATION OF {image_size}x{image_size} took on avg: {optimization_benchmark(image_size)}')
        print(f'PPN+EFFECT OF {image_size}x{image_size} took on avg: {ppn_benchmark(image_size)}')
