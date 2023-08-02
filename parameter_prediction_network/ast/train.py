from pathlib import Path

from parameter_prediction_network.ast.ast_ppn_model import AST_PPN_MODEL
from parameter_prediction_network.model_helper import setup_and_start_training, get_model_argparser, add_train_args

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
    parser.add_argument('--effect', help='The name of the effect that should be used.', default="arbitrary_style")
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
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Defines the content weight for the sanet loss.')
    parser.add_argument('--style_weight', type=float, default=3.0,
                        help='Defines the content weight for the sanet loss.')
    parser.add_argument('--wikiart_dir', type=str, required=True,
                        help='Folder containing the wikiart style images.')

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    base_dir = Path(args.logs_dir) / 'SANet'

    model = AST_PPN_MODEL(args, base_dir)
    setup_and_start_training(args, model, base_dir)
