import random
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from effects.arbitrary_style_first_stage import perform_first_stage_segmentation
from parameter_prediction_network.ast.ast_network import SANetForNST
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel

# Augmentation options
MIN_SIGMA = 1.0
MAX_SIGMA = 1.0

KERNEL_SIZES = [3]  # [1, 3, 5, 7]

BASE_NUM_SEGMENTS_PER_PIXEL = 5000 / (512 * 409)

NUM_SEGMENTS_OFFSET_MIN = 1.0
NUM_SEGMENTS_OFFSET_MAX = 1.0


def get_random_first_stage_params(image):
    sigma = random.uniform(MIN_SIGMA, MAX_SIGMA)
    kernel_size = random.choice(KERNEL_SIZES)
    n_segments_factor = random.uniform(NUM_SEGMENTS_OFFSET_MIN, NUM_SEGMENTS_OFFSET_MAX)
    n_segments = int(BASE_NUM_SEGMENTS_PER_PIXEL * image.shape[-2] * image.shape[-1] * n_segments_factor)
    return kernel_size, n_segments, sigma


class JohnsonNSTTransform(torch.nn.Module):
    def __init__(self, nst_path: str, device: Any):
        super().__init__()
        self.device = device
        self.nst_model = None
        self.nst_path = nst_path

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.nst_model is None:
            self.nst_model = torch.jit.script(
                JohnsonNSTModel.load_from_checkpoint(self.nst_path, strict=False).network).to(self.device)
        kernel_size, n_segments, sigma = get_random_first_stage_params(image)
        with torch.no_grad():
            orig_size = image.shape[-2:]
            nst_img = self.nst_model(image.unsqueeze(dim=0).to(self.device))
            nst_img = torch.sigmoid(nst_img)
            nst_img = torch.nn.functional.interpolate(nst_img, orig_size)
            nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
            nst_image, segmentation_labels, segmented_img = perform_first_stage_segmentation(nst_img, kernel_size,
                                                                                             sigma, n_segments)
        return segmented_img[0].cpu(), image.cpu(), nst_image[0].cpu()


class SANetNSTTransform(torch.nn.Module):
    def __init__(self, wikiart_dir: str, device: Any):
        super().__init__()
        self.device = device
        self.nst_model = None
        self.style_transforms = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ])
        self.style_imgs = [str(style_img) for style_img in Path(wikiart_dir).iterdir()]
        self.style_img_idx = 0
        self.style_permutation = np.random.permutation(len(self.style_imgs))

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.nst_model is None:
            self.nst_model = torch.jit.script(SANetForNST()).to(self.device)
        kernel_size, n_segments, sigma = get_random_first_stage_params(image)

        with Image.open(self.style_imgs[self.style_permutation[self.style_img_idx]]) as style_img:
            style_img = style_img.convert('RGB')
            style_img = self.style_transforms(style_img).to(self.device).unsqueeze(0)
        self.style_img_idx += 1
        if self.style_img_idx >= len(self.style_imgs):
            self.style_img_idx = 0
            self.style_permutation = np.random.permutation(len(self.style_imgs))

        with torch.no_grad():
            orig_size = image.shape[-2:]
            nst_img = self.nst_model(image.unsqueeze(0).to(self.device), style_img)
            nst_img = torch.nn.functional.interpolate(nst_img, orig_size)
            nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
            nst_image, segmentation_labels, segmented_img = perform_first_stage_segmentation(nst_img, kernel_size,
                                                                                             sigma, n_segments)
        return segmented_img[0].cpu(), image.cpu(), style_img[0].cpu()


def get_dataloader_device(args):
    if getattr(args, 'effect', None) != 'arbitrary_style':
        return 'cpu'
    assert args.num_train_gpus < torch.cuda.device_count(), f'Found {torch.cuda.device_count()} gpus and are ' \
                                                            f'using the first {args.num_train_gpus} gpus for ' \
                                                            f'training model but need the last gpu for dataloading.'
    device = torch.device('cuda', torch.cuda.device_count() - 1)
    # device = torch.device('cpu')
    print(f'Using {device} for dataloading.')
    return device
