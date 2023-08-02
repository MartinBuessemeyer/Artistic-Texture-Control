import argparse
from pathlib import Path

from parameter_prediction_network.model_base import ModelBase


class JohnsonNSTModel(ModelBase):
    def __init__(self, args: argparse.Namespace, base_dir: Path):
        super().__init__(args, base_dir, 3, (0.0, 1.0))
        self.save_hyperparameters()
