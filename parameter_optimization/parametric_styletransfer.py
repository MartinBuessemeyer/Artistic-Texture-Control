import argparse
from pathlib import Path

import imageio
import torch
import torch.nn.functional as F

from brushstroke.paint_transformer.inference.image_painter import BrushstrokeType, FirstStageImagePainter
from effects import get_default_settings
from effects.arbitrary_style_first_stage import perform_first_stage_segmentation, perform_first_stage_painter, \
    apply_blur
from effects.gauss2d_xy_separated import Gauss2DEffect
from helpers import load_image_cuda, save_as_image
from helpers.effect_base import EffectBase
from helpers.losses import CLIPstylerLoss, FlippedVgg16Loss, StrottsLoss
from parameter_optimization.optimization_utils import run_optimization_loop, ParameterContainer
from parameter_optimization.strotss_org import execute_style_transfer
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel


def single_optimize(effect: EffectBase, preset_or_vp, loss, input, target,
                    args, iter_callback=lambda step: None):
    experiment_dir = get_experiment_folder(args)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    output_name = f"{experiment_dir}/OUT_{loss.__class__.__name__}"
    save_as_image(target, experiment_dir / 'target_resized.png')

    if isinstance(preset_or_vp, torch.Tensor):
        vp = preset_or_vp
    elif isinstance(preset_or_vp, list):  # preset
        vp = effect.vpd.preset_tensor(preset_or_vp, input, not args.use_global_parameters)

    # if isinstance(effect, OilPaintEffect) or isinstance(effect, WatercolorEffect):
    # effect.preprocess = AdaptHueSelfManagingParameterEffect(input, smooth=False)
    # module.postprocess = AdaptHueSelfManagingParameterEffect(batch_im, smooth=False)

    grad_vp = ParameterContainer(vp, smooth=False)
    writer = imageio.get_writer(f"{output_name}_video.mp4", fps=30) if args.generate_video else None
    gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=5)
    gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=5)

    def cbck(loss, out, lr, i):
        if i in args.smoothing_steps:
            # smooth the parameters every few iterations
            # this should decrease artifacts
            vp_smoothed = gauss2dx(grad_vp.vp.data, torch.tensor(args.sigma_large).cuda())
            grad_vp.vp.data = gauss2dy(vp_smoothed, torch.tensor(args.sigma_large).cuda())
        iter_callback(i)

    result, _ = run_optimization_loop(effect, grad_vp, input, target, loss, output_name, args,
                                      vid=writer, callback=cbck, verbose=True)

    if writer is not None:
        writer.close()

    save_as_image(result, f"{output_name}.png", clamp=True)
    save_as_image(input, f"{output_name}_input.png", clamp=True)

    # save the parameter maps
    xxx = grad_vp()
    torch.save(xxx.detach().clone(), f"{output_name}.pt")
    return xxx, input


def generate_strotss_result(s, t,
                            experiment_dir):
    experiment_dir.mkdir(exist_ok=True, parents=True)
    strotss_out = experiment_dir / ("strotts_" + Path(s).name)

    if not Path(strotss_out).exists():
        result = execute_style_transfer(s, t, resize_to=1024)
        result.save(strotss_out)
    return strotss_out


def get_experiment_folder(args: argparse.Namespace) -> Path:
    exp_name = args.experiment_name
    return Path(args.output_dir) / exp_name


def get_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', help='content image', required=True, type=str)
    parser.add_argument('--style', help='style image',
                        default=None, type=str)
    parser.add_argument('--img_size', help='ouptut image size',
                        default=512, type=int)
    parser.add_argument('--output_dir', help='output directory in results', default="experiments")
    parser.add_argument('--experiment_name', help='Makes up the output path combined with --output_dir.',
                        default='test')
    parser.add_argument('--use_gradient_checkpointing', help='Activates gradient checkpointing. '
                                                             'Reduces consumed GPU memory at some performace costs.',
                        action='store_true')
    parser.add_argument('--generate_video', help='Generates a video of the optimization process of the result. ',
                        action='store_true')
    parser.add_argument('--loss', help='Define the used loss function. Default L1 loss.',
                        choices=['L1', 'CLIPStyler', 'GatysLoss', 'STROTTS'],
                        default='L1')
    parser.add_argument("--nst_variant", choices=['STROTTS', 'JohnsonNST'], default='STROTTS',
                        help='Which NST should be used to generate target image.')
    parser.add_argument("--johnson_nst_checkpoint", default=None, type=str,
                        help='Path to the johnson nst checkpoint, used when --nst_variant is johnson_nst.')

    # arbitrary_style pipeline options
    parser.add_argument("--first_stage_type", choices=['PaintTransformer', 'Segmentation'], default='Segmentation',
                        help='Which version should be used for the first pipeline stage of the arbitrary style pipeline.')
    parser.add_argument("--first_stage_sigma", default=1.0, type=float,
                        help='Global smoothing sigma at end of first pipeline stage of arbitrary style pipeline.')
    parser.add_argument("--first_stage_kernel_size", default=3, type=int,
                        help='Global smoothing kernel size at end of first pipeline stage of arbitrary style pipeline.')
    parser.add_argument("--n_segments", help='Number of segments when using the segmentation as '
                                             '--first_stage_type for arbitary style pipeline.', default=5000, type=int)

    # Regional consistency loss options
    parser.add_argument("--rc_brushstroke_weight", help='Set to != 0 to apply the brushstroke loss component '
                                                        'of the regional consistency loss to the parameter masks.'
                                                        'High performance costs.',
                        default=0.0, type=float)
    parser.add_argument("--rc_tv_weight", help='Set to != 0 to apply the total variation loss component '
                                               'of the regional consistency loss to the parameter masks.'
                                               'Almost no performance costs.',
                        default=0.0, type=float)

    # CLIP loss options
    parser.add_argument('--clipstyler_text', help='Use Clip loss for optimization with given text prompt.',
                        default=None)
    parser.add_argument('--clipstyler_content_weight', help='Sets the content weight of the CLIPStyler loss.',
                        default=150, type=float)
    parser.add_argument('--clipstyler_crop_size', help='Sets the crop size of the CLIPStyler loss.',
                        default=128, type=int)

    # Optimization process options
    parser.add_argument('--lr_start', help='Learning at the beginning of optimization.',
                        default=0.02, type=float)
    parser.add_argument('--lr_stop', help='Minimal learning during optimization.',
                        default=0.02, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decaying factor.',
                        default=0.99, type=float)
    parser.add_argument('--lr_decay_start', help='At which optimization step the learning rate decay starts.',
                        default=500, type=int)
    parser.add_argument('--n_iterations', help='Number of optimization steps to perform.',
                        default=500, type=int)
    parser.add_argument('--grad_clip', help='Activates gradient clipping when set to != 0.0.',
                        default=0.0, type=float)
    parser.add_argument('--smoothing_steps', nargs='+',
                        help='Apply gaussian smoothing of parameter masks at specified iterations.'
                             'Only recommended for watercolor pipeline optimization: 10 25 50 100 250 500.',
                        default=[], type=int)
    parser.add_argument('--use_global_parameters', help='Optimize global parameters, i.e. one value per '
                                                        'parameter (no parameter masks). '
                                                        'Will lead to inferior results.', action='store_true')
    parser.add_argument('--sigma_large',
                        help='Sigma value for gaussian smoothing of parameter masks at specified iterations.'
                             'Only recommended for watercolor pipeline optimization.',
                        default=1.5, type=float)
    return parser


def raise_exception_on_invalid_configuaration(args: argparse.Namespace):
    if args.loss == 'CLIPStyler' and not args.clipstyler_text:
        raise ValueError('You need to specify a text prompt when using the CLIPStyler loss.')
    if args.nst_variant == 'JohnsonNST' and not args.johnson_nst_checkpoint:
        raise ValueError(
            'You are attempting to use the Johnson NST to generate a target image instead of the default '
            'STROTTS method. You need to supply the weights for the Johnson NST.')
    if args.nst_variant == 'strotts' and not args.style:
        raise ValueError('The STROTTS NST method requires a style image. Please set --style.')


def get_loss_function(args: argparse.Namespace, content: torch.Tensor, input_img: torch.Tensor) -> torch.nn.Module:
    if args.loss == 'L1':
        loss = torch.nn.L1Loss()
    elif args.loss == 'CLIPStyler':
        loss = CLIPstylerLoss(content_weight=args.clipstyler_content_weight, crop_size=args.clipstyler_crop_size)
        loss.set_raw_content_image(content)
        loss.set_source(input_img)
        loss.set_text_prompt(args.clipstyler_text)
    elif args.loss == 'GatysLoss':
        loss = FlippedVgg16Loss(args.content, 256, args.style, image_dim=256, style_weight=1e11)
    elif args.loss == 'STROTTS':
        loss = StrottsLoss(args.content, args.style, content_weight=8.0)
    else:
        raise ValueError('Attemting to use invalid loss.')
    return loss


def get_optim_related_imgs_imgs(args: argparse.Namespace) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    content = load_image_cuda(args.content, long_edge=args.img_size)
    target = get_target_img(args, content)
    input_img = F.interpolate(get_input_img(args, content, target), (target.size(2), target.size(3)))
    return content, input_img, target


def get_target_img(args: argparse.Namespace, content: torch.Tensor) -> torch.Tensor:
    if args.nst_variant not in ['STROTTS', 'JohnsonNST']:
        raise ValueError("Attempting to use invalid nst_variant.")
    if args.nst_variant == 'STROTTS':
        experiment_dir = get_experiment_folder(args)
        strotts_path = generate_strotss_result(args.content, args.style, experiment_dir)
        return load_image_cuda(strotts_path, long_edge=args.img_size)
    elif args.nst_variant == 'JohnsonNST':
        return JohnsonNSTModel.load_from_checkpoint(args.johnson_nst_checkpoint).predict_keep_img_size(content).detach()


def get_input_img(args: argparse.Namespace, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if args.first_stage_type == 'segmentation':
        _, _, segmented_img = perform_first_stage_segmentation(target, 0, 0.0, args.n_segments)
    elif args.first_stage_type == 'PaintTransformer':
        segmented_img = perform_first_stage_painter(target, FirstStageImagePainter(BrushstrokeType.CIRCLE).to('cuda'),
                                                    0, 0.0)
    else:
        raise ValueError("Invalid --first_stage_type.")
    segmented_img = apply_blur(segmented_img, args.first_stage_kernel_size, args.first_stage_sigma)
    save_as_image(segmented_img, get_experiment_folder(args) / "first_stage_output.png", clamp=True)
    return segmented_img


if __name__ == '__main__':
    parser = get_options()
    args = parser.parse_args()
    args.effect = "arbitrary_style"
    raise_exception_on_invalid_configuaration(args)
    effect, preset, _ = get_default_settings(args.effect)
    if args.use_gradient_checkpointing:
        effect.enable_checkpoints()
    content, input_img, target = get_optim_related_imgs_imgs(args)
    single_optimize(effect.cuda(), preset, get_loss_function(args, content, input_img).cuda(), input_img, target, args)
