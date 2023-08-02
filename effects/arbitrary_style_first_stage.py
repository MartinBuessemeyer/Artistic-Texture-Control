from typing import Tuple, Optional, Dict

import skimage
import torch
import torch_scatter
import torchvision

from brushstroke.paint_transformer.inference.image_painter import FirstStageImagePainter


def perform_first_stage_segmentation(nst_image: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0,
                                     n_segments=5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        '''
        # paint transformer
        # scale up image for better painting quality.
        scaling_factor = 4
        scaled_image = torch.nn.functional.interpolate(batch_im, (
            image.size(2) * scaling_factor, image.size(3) * scaling_factor))
        nst_img = nst_model(scaled_image)
        nst_img = torch.nn.functional.interpolate(nst_img, (image.size(2), image.size(3)))
        '''
        # segmentation painter
        segmentation_labels = segment_image(nst_image, n_segments)
        segmented_img = render_labels(segmentation_labels, nst_image, kernel_size, sigma)
        return nst_image, segmentation_labels, segmented_img


def perform_first_stage_painter(nst_image: torch.Tensor, painter: torch.nn.Module, kernel_size: int = 3,
                                sigma: float = 1.0) -> torch.Tensor:
    painted_image = painter(nst_image)
    return apply_blur(painted_image, kernel_size, sigma)


# TODO breaks a lot with the detailed mask.
def perform_first_stage_paint_transformer(stylized_input: torch.Tensor, paint_transformer: FirstStageImagePainter,
                                          details_mask: torch.Tensor, prescale_factor: float = 1.5):
    scaled_image = torch.nn.functional.interpolate(stylized_input, (
        int(stylized_input.size(2) * prescale_factor), int(stylized_input.size(3) * prescale_factor)))
    # pad scaled image by 15 percent with reflection padding
    pad = int(scaled_image.size(2) * 0.15)
    scaled_image = torch.nn.functional.pad(scaled_image, (pad, pad, pad, pad), mode='reflect')

    segmented_image = paint_transformer(scaled_image, details_mask)
    # crop 15 percent from each side
    segmented_image = segmented_image[:, :, pad:int(segmented_image.size(2) - pad),
                      pad:int(segmented_image.size(3) - pad)]

    segmented_image = torch.nn.functional.interpolate(segmented_image, (stylized_input.size(2), stylized_input.size(3)))

    # segmented_image = paint_transformer(stylized_input, details_mask)
    black_cutoff = 120.0 / 255.0
    # find darkest color that is not completely black in segmented_image
    black_spots_whitened = torch.where(segmented_image.sum(dim=1) < black_cutoff, torch.ones_like(segmented_image),
                                       segmented_image)
    darkest_color = torch.min(black_spots_whitened.view(1, 3, -1), dim=2, keepdim=False)[0]
    darkest_color = darkest_color.view(1, 3, 1, 1).repeat(1, 1, segmented_image.shape[2], segmented_image.shape[3])
    # fills all black spots with the darkest color
    segmented_image = torch.where(segmented_image.sum(dim=1) < black_cutoff, darkest_color, segmented_image)

    return stylized_input, segmented_image


def segment_image(nst_image: torch.Tensor, n_segments: int, min_size_factor: float = 0.25) -> torch.Tensor:
    return torch.from_numpy(
        skimage.segmentation.slic(nst_image.permute(0, 2, 3, 1)[0].cpu().numpy(), n_segments=n_segments,
                                  # compactness=0.00000001, max_num_iter=500,
                                  min_size_factor=min_size_factor, convert2lab=True, slic_zero=True)).to(
        nst_image.device)[None, None, :]
    '''return torch.from_numpy(
        cuda_slic(nst_image.permute(0, 2, 3, 1)[0].cpu().numpy(), convert2lab=True, n_segments=n_segments,
                  min_size_factor=min_size_factor,
                  compactness=10, max_iter=10)).to(nst_image.device)[None, None, :]'''


def perform_first_stage_with_nst(image: torch.Tensor, nst_model: torch.nn.Module, kernel_size: int = 3,
                                 sigma: float = 1.0, n_segments=5000) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        orig_size = image.shape[-2:]
        nst_img = nst_model(image)
        nst_img = torch.nn.functional.interpolate(nst_img, orig_size)
        nst_img = torch.clamp(nst_img, min=0.0, max=1.0)
        return perform_first_stage_segmentation(nst_img, kernel_size, sigma, n_segments)


def render_labels(segmentation_labels: torch.Tensor, nst_img: torch.Tensor, kernel_size: int = 3,
                  sigma: float = 1.0) -> torch.Tensor:
    segmented_img = _labels_to_avg_color_scatter(segmentation_labels, nst_img)
    return apply_blur(segmented_img, kernel_size, sigma)


def apply_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1 or sigma <= 0.0:
        return img
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blurrer(img)


def _labels_to_avg_color_pytorch(labels: torch.tensor, image: torch.tensor, base_image: Optional[torch.Tensor],
                                 label_to_custom_color: Dict[int, Tuple[int, int, int]]) -> torch.tensor:
    out = torch.zeros(image.shape, dtype=torch.float, device=image.device) if base_image is None else base_image
    unique_labels = torch.unique(labels[labels != 0], sorted=False)
    for label in unique_labels:
        mask = (labels == label)
        # color = image[mask].mean(dim=0)
        color = torch.median(image[mask], dim=0).values if label_to_custom_color.get(label.item(), None) is None \
            else torch.tensor(label_to_custom_color[label.item()], device=image.device)
        out[mask] = color
    return out


@torch.jit.script
def _labels_to_avg_color_scatter(labels: torch.Tensor, image: torch.Tensor):
    # labels = torch.from_numpy(labels).to(device).long()
    # image = torch.from_numpy(image).to(device) / 255.0
    _, c, h, w = image.size()
    mean_color = torch_scatter.scatter(image.view(c, h * w), labels.view(-1), reduce="mean")
    seg_out = mean_color[:, labels.view(-1)]
    return seg_out.view((1, c, h, w))
