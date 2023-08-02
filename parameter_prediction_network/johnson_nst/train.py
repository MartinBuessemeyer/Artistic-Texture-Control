from pathlib import Path

from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel
from parameter_prediction_network.model_helper import setup_and_start_training, get_model_argparser, add_train_args

'''
The following script learns a local parameter prediction network for a specified effect and style.
Example call:
```
python -m parameter_prediction_network.johnson_nst.train --dataset_path /data/train2017/ --batch_size 8 \
 --logs_dir ./logs --style ./experiments/target/oilpaint_portrait.jpg --name experiment_name
```
The command line arguments are explained in the argparse help.
'''

if __name__ == '__main__':
    args = add_train_args(get_model_argparser()).parse_args()
    base_dir = Path(args.logs_dir) / 'johnson_nst'
    model = JohnsonNSTModel(args, base_dir)
    setup_and_start_training(args, model, base_dir)
