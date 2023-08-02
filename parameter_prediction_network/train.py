from pathlib import Path

from parameter_prediction_network.model_helper import setup_and_start_training, get_model_argparser, add_train_args
from parameter_prediction_network.ppn_model import PPNModel

'''
The following script learns a local parameter prediction network for a specified effect and style.
Example call:
```
python -m parameter_prediction_network.train --dataset_path /data/train2017/ --batch_size 8 \
 --logs_dir ./logs --style ./experiments/target/oilpaint_portrait.jpg
```
The command line arguments are explained in the argparse help.
'''


def get_argparser():
    parser = get_model_argparser()
    add_train_args(parser)

    # PPN related args
    parser.add_argument('--enable_adapt_hue_preprocess', help='Enables the AdaptHueEffect as a preprocessing step'
                                                              'if supported by the used effect.',
                        action='store_true')
    parser.add_argument('--enable_adapt_hue_postprocess', help='Enables the AdaptHueEffect as a postprocessing step'
                                                               'if supported by the used effect.',
                        action='store_true')
    parser.add_argument('--enable_depth_enhance',
                        help='Enables the depth enhance pass at the beginning of the effect.',
                        action='store_true')
    parser.add_argument('--disable_gradient_checkpoints', help='Disables gradient checkpointing for the effect. '
                                                               'This trades Runtime for RAM. '
                                                               'Note that training might crash randomly when activated.'
                                                               'This happens, when all gradients of a checkpoint are '
                                                               'None or zero.', action='store_true')
    parser.add_argument('--brushstroke_weight', type=float, default=0.0, help='Defines the weight of the '
                                                                              'brushstroke loss applied on '
                                                                              'the parameter masks. '
                                                                              'Use 0.0 to disable.')
    parser.add_argument('--disable_jit_brushstroke', action='store_true', help='Deactivates torch.jit for brushstroke'
                                                                               'loss.')
    parser.add_argument('--mask_tv_weight', type=float, default=0.0, help='Defines the weight of the tv loss '
                                                                          'appield on the parameter masks. '
                                                                          'Use 0.0 to disable.')
    parser.add_argument('--johnson_nst_model', type=str, default=None, help='Path to trained johnson nst '
                                                                            'model for first stage of '
                                                                            'arbitrary pipeline.')

    parser.add_argument('--use_l1_loss', action='store_true', help='Use l1 loss for optimization instead of '
                                                                   'perceptual_loss. Only supported with '
                                                                   'arbitrary style pipeline.')

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.effect = "arbitrary_style"
    base_dir = Path(args.logs_dir) / args.effect

    if args.use_l1_loss and args.effect != 'arbitrary_style':
        print('L1 Loss only supported for arbitrary style pipeline.')
        exit(-1)

    model = PPNModel(args, base_dir)
    setup_and_start_training(args, model, base_dir)
