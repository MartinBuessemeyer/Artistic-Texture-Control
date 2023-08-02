import argparse
from typing import Dict

import torch
from PIL import Image

from helpers import np_to_torch
from helpers.losses import PerceptualStyleLoss, PerceptualContentLoss


def load_image(path: str) -> torch.Tensor:
    return np_to_torch(Image.open(path).convert("RGB"))


def get_metrics(y: torch.Tensor, y_hat: torch.Tensor, style_image_path: str,
                content_image: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {'style_loss': PerceptualStyleLoss(style_image_path)(y_hat),
               'content_loss': PerceptualContentLoss()(y_hat, content_image),
               'l1_loss': torch.nn.functional.l1_loss(y, y_hat)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--y', help='Ground Truth Image path.', type=str)
    parser.add_argument('--y_hat', help='Predicted Image path.', type=str)
    parser.add_argument('--style_image', help='Path to the style image.', type=str)
    parser.add_argument('--content_image', help='Path to the content image.', type=str)
    args = parser.parse_args()

    y = load_image(args.y)
    y_hat = load_image(args.y_hat)
    content_image = load_image(args.content_image)

    print(get_metrics(y, y_hat, args.style_image, content_image))
