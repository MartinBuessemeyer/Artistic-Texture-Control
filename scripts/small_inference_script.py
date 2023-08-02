from pathlib import Path

import torch
from torchvision import transforms

from brushstroke.paint_transformer.inference.image_painter import FirstStageImagePainter, BrushstrokeType
from effects.arbitrary_style_first_stage import perform_first_stage_segmentation, perform_first_stage_painter
from helpers import load_image_cuda, save_as_image
from parameter_optimization.strotss_org import execute_style_transfer
from parameter_prediction_network.ast.ast_network import SANetForNST
from parameter_prediction_network.ast.ast_ppn_model import AST_PPN_MODEL
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel
from parameter_prediction_network.ppn_model import PPNModel

NUMBER = 4
STYLE_NAME = 'delaunay'
SEC_STYLE_NAME = 'mosaic'

BASE_DIR = Path('./path/to/dir')
CONTENT_IMG = BASE_DIR / f'content_{NUMBER}.jpg'
STYLE_IMG = BASE_DIR / f'style_{NUMBER}.jpg'
DEST_DIR = BASE_DIR

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

STYLE_TO_OILPAINT_PPN = {
    'candy': './logs/oilpaint/checkpoints/oilpaint_finetune/candy_arch_jin_oil_hue_pre_depth_sw_7.5e+09_256/last.ckpt',
    'delaunay': './logs/oilpaint/checkpoints/oilpaint_finetune/delaunay_arch_jin_oil_hue_pre_depth_sw_7.5e+10_256/last-v1.ckpt',
    'Femme_nue_assise': './logs/oilpaint/checkpoints/oilpaint_finetune/Femme_nue_assise_arch_jin_oil_hue_pre_depth_sw_1.0e+11_256/last-v1.ckpt',
    'mosaic': './logs/oilpaint/checkpoints/oilpaint_finetune/mosaic_arch_jin_oil_hue_pre_depth_256/last.ckpt',
    'rain-princess-cropped': './logs/oilpaint/checkpoints/oilpaint_finetune/rain-princess-cropped_arch_jin_oil_hue_pre_depth_256/last.ckpt',
    'starry_night': './logs/oilpaint/checkpoints/oilpaint_finetune/starry_night_arch_jin_oil_hue_pre_depth_sw_5.0e+10_256/last-v3.ckpt',
    'the_scream': './logs/oilpaint/checkpoints/oilpaint_finetune/the_scream_arch_jin_oil_hue_pre_depth_sw_5.0e+10_256/last-v1.ckpt',
    'the_shipwreck_of_the_minotaur': './logs/oilpaint/checkpoints/oilpaint_finetune/the_shipwreck_of_the_minotaur_arch_jin_oil_hue_pre_depth_sw_7.5e+10_256/last.ckpt',
    'udnie': './logs/oilpaint/checkpoints/oilpaint_finetune/udnie_arch_jin_oil_hue_pre_depth_sw_2.5e+10_256/last-v1.ckpt',
    'wave': './logs/oilpaint/checkpoints/oilpaint_finetune/wave_arch_jin_oil_hue_pre_depth_256/last.ckpt'}

SANET_PPN_PATH = Path('./logs/SANet/checkpoints/train/sanet_lr_1e5/last.ckpt')

IMG_SIZE = 512
KERNEL_SIZE = 3
SIGMA = 1.0
NUM_SEGMENTS = 5000


def original():
    style_img = load_image_cuda(STYLE_IMG, 512)
    style_transforms = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
    ])
    style_img = style_transforms(style_img)

    content_img = load_image_cuda(CONTENT_IMG)
    out_shape = content_img.shape[-2:]

    ast_nst = SANetForNST().cuda()
    ast_ppn = AST_PPN_MODEL.load_from_checkpoint(SANET_PPN_PATH).cuda()

    DIVIDABLE = 16
    orig_size = list(out_shape)
    orig_size[0] = orig_size[0] - (orig_size[0] % DIVIDABLE)
    orig_size[1] = orig_size[1] - (orig_size[1] % DIVIDABLE)
    content_img_resize = torch.nn.functional.interpolate(content_img, orig_size)

    nst_img = ast_nst(content_img_resize, style_img)
    nst_img = torch.nn.functional.interpolate(nst_img, orig_size)
    nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

    nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, KERNEL_SIZE, SIGMA, NUM_SEGMENTS)

    ast_stylized = ast_ppn(segmented_img, style_img)
    ast_stylized = torch.nn.functional.interpolate(ast_stylized, out_shape)
    save_as_image(ast_stylized, DEST_DIR / f'ast_{NUMBER}.png')

    johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[STYLE_NAME]).cuda()
    ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[STYLE_NAME], strict=False).cuda()

    nst_img = johnson_nst(content_img)
    # nst_img = torch.nn.functional.interpolate(nst_img, out_shape)
    # nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

    nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, KERNEL_SIZE, SIGMA, NUM_SEGMENTS)

    ppn_stylized = ppn(segmented_img)
    ppn_stylized = torch.nn.functional.interpolate(ppn_stylized, out_shape)
    save_as_image(ppn_stylized, DEST_DIR / f'ppn_{NUMBER}.png')


def ppn_comparison():
    content_img = load_image_cuda(CONTENT_IMG)

    oilpaint_nst = PPNModel.load_from_checkpoint(STYLE_TO_OILPAINT_PPN[STYLE_NAME]).cuda()
    save_as_image(oilpaint_nst(content_img), DEST_DIR / f'oilpaint_{NUMBER}.png')

    johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[STYLE_NAME]).cuda()
    ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[STYLE_NAME], strict=False).cuda()

    nst_img = johnson_nst(content_img)
    # nst_img = torch.nn.functional.interpolate(nst_img, out_shape)
    # nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

    nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, n_segments=NUM_SEGMENTS)

    ppn_stylized = ppn(segmented_img)
    save_as_image(ppn_stylized, DEST_DIR / f'as_{NUMBER}.png')


def ppn_pt_first_stage():
    content_img = load_image_cuda(CONTENT_IMG)

    johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[STYLE_NAME]).cuda()
    ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[STYLE_NAME], strict=False).cuda()

    nst_img = johnson_nst(content_img)
    # nst_img = torch.nn.functional.interpolate(nst_img, out_shape)
    # nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

    segmented_img = perform_first_stage_painter(nst_img,
                                                painter=FirstStageImagePainter(BrushstrokeType.BRUSH).to('cuda'))
    save_as_image(segmented_img, DEST_DIR / f'pt_{NUMBER}.png')

    ppn_stylized = ppn(segmented_img)
    save_as_image(ppn_stylized, DEST_DIR / f'mixed_{NUMBER}.png')


def mixing_styles():
    content_img = load_image_cuda(CONTENT_IMG)

    johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[STYLE_NAME]).cuda()
    ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[STYLE_NAME], strict=False).cuda()
    mixed_ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[SEC_STYLE_NAME], strict=False).cuda()

    nst_img = johnson_nst(content_img)
    # nst_img = torch.nn.functional.interpolate(nst_img, out_shape)
    # nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

    nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, n_segments=NUM_SEGMENTS)

    save_as_image(ppn(segmented_img), DEST_DIR / f'same_{NUMBER}.png')
    save_as_image(mixed_ppn(segmented_img), DEST_DIR / f'mixed_{NUMBER}.png')


def ppn_examples():
    content_img = load_image_cuda(CONTENT_IMG)

    johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[STYLE_NAME]).cuda()
    ppn: PPNModel = PPNModel.load_from_checkpoint(STYLE_TO_PPN[STYLE_NAME], strict=False).cuda()

    nst_img = johnson_nst(content_img)
    nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
    save_as_image(nst_img, DEST_DIR / f'sst_{NUMBER}.png')
    # nst_img = torch.nn.functional.interpolate(nst_img, out_shape)
    # nst_img = torch.clamp(nst_img, min=0.0, max=1.0)

    nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, n_segments=NUM_SEGMENTS)

    save_as_image(ppn(segmented_img), DEST_DIR / f'ppn_sst_{NUMBER}.png')

    style_img = load_image_cuda(STYLE_IMG, 512)
    style_transforms = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
    ])
    style_img = style_transforms(style_img)

    ast_nst = SANetForNST().cuda()
    ast_ppn = AST_PPN_MODEL.load_from_checkpoint(SANET_PPN_PATH).cuda()

    nst_img = ast_nst(content_img, style_img)
    nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
    save_as_image(nst_img, DEST_DIR / f'sanet_{NUMBER}.png')

    nst_image, _, segmented_img = perform_first_stage_segmentation(nst_img, KERNEL_SIZE, SIGMA, NUM_SEGMENTS)

    ast_stylized = ast_ppn(segmented_img, style_img)
    save_as_image(ast_stylized, DEST_DIR / f'ppn_ast_{NUMBER}.png')


def rw_examples():
    content_img = load_image_cuda(CONTENT_IMG)

    johnson_nst = JohnsonNSTModel.load_from_checkpoint(STYLE_TO_JOHNSON_NST[STYLE_NAME]).cuda()

    nst_img = johnson_nst(content_img)
    nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
    save_as_image(nst_img, DEST_DIR / f'sst_{NUMBER}.png')

    style_img = load_image_cuda(STYLE_IMG, 512)
    style_transforms = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
    ])
    style_img = style_transforms(style_img)

    ast_nst = SANetForNST().cuda()

    nst_img = ast_nst(content_img, style_img)
    nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
    save_as_image(nst_img, DEST_DIR / f'sanet_{NUMBER}.png')


def strotts():
    result = execute_style_transfer(CONTENT_IMG, STYLE_IMG, 512)
    result.save(DEST_DIR / f'strotts_{NUMBER}.png')


if __name__ == '__main__':
    # original()
    # ppn_comparison()
    # ppn_pt_first_stage()
    # mixing_styles()
    # ppn_examples()
    # rw_examples()
    strotts()
