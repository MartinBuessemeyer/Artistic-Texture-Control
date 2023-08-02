import argparse
from pathlib import Path

from effects import get_effect_short_name
from parameter_prediction_network.ppn_architectures import PPNArchitectures


def args_to_train_name(args: argparse.Namespace) -> str:
    name = f'{Path(args.style).stem}_arch_{PPNArchitectures.to_log_name(args.architecture)}_'
    if not hasattr(args, "effect"):
        name += 'johnson_nst'
    else:
        name += f'{get_effect_short_name(args.effect)}'
        if args.enable_adapt_hue_preprocess or args.enable_adapt_hue_postprocess:
            name += f'_hue{"_pre" if args.enable_adapt_hue_preprocess else ""}' \
                    f'{"_post" if args.enable_adapt_hue_postprocess else ""}'
        if args.enable_depth_enhance:
            name += '_depth'
    if args.use_vgg19_loss:
        name += '_vgg19'
    if args.style_weight != 1e10:
        name += f'_sw_{args.style_weight:.1e}'
    if args.tv_weight:
        name += f'_tv_{args.tv_weight:.1e}'
    if args.grad_clip:
        name += f'_grad_clip_{args.grad_clip:.1e}'
    if args.style_img_size:
        name += f'_{args.style_img_size}'
    return name


def get_exp_name_and_interm_folder(args, base_dir):
    experiment_name = args_to_train_name(args) if not args.experiment_name else args.experiment_name
    (base_dir / 'configs').mkdir(parents=True, exist_ok=True)
    with open(base_dir / 'configs' / f'{experiment_name}.txt', 'wt') as config_file:
        config_file.write(str(args).replace(' ', '\n'))
    intermediate_results_folder = base_dir / 'intermediate_result_imgs' / args.group_name / experiment_name
    intermediate_results_folder.mkdir(parents=True, exist_ok=True)
    return experiment_name, intermediate_results_folder
