import pickle
import random
import string
from typing import Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from matplotlib.colors import to_rgb
from skimage import exposure
from torch.nn import functional as F

from helpers import save_as_image, torch_to_np
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel

SELECTION_COLOR = np.array([255, 255, 255])  # white
SEC_SELECTION_COLOR = np.array([0, 0, 0])  # black


def hex_color_to_int_tuple(hex_str: str) -> Tuple[int, int, int]:
    r, g, b = to_rgb(hex_str)
    return int(r * 255), int(g * 255), int(b * 255)


def load_nst_model(johnson_nst_checkpoint_path):
    return JohnsonNSTModel.load_from_checkpoint(johnson_nst_checkpoint_path, strict=False)


def load_vps(vp_path, content_image, base_dir, state, effect, preset) -> torch.Tensor:
    if vp_path:
        vp = torch.load(vp_path).detach().clone()
        vp = F.interpolate(vp, content_image.shape[2:])

        for i in range(vp.size(1)):
            (base_dir / "vps").mkdir(exist_ok=True)
            save_as_image(vp[:, i], base_dir / 'vps' / f"{i}.png")

        torch.save(vp, base_dir / "vp_buffer.pt")
        state.file_load_key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        return vp
    elif (base_dir / "vp_buffer.pt").exists():
        return torch.load(base_dir / "vp_buffer.pt")
    else:
        vp = effect.vpd.preset_tensor(preset, content_image, add_local_dims=True)
        torch.save(vp, base_dir / "vp_buffer.pt")
        return vp


# st.cache
def np_to_display_image(image: np.array) -> Image.Image:
    return Image.fromarray((image.squeeze() * 255.0).astype(np.uint8))


# st.cache
def torch_to_display_image(tensor: torch.Tensor) -> Image.Image:
    return np_to_display_image(torch_to_np(tensor))


def add_interpolation_mask(mask: torch.Tensor, inverted: bool, base: torch.Tensor) -> torch.Tensor:
    mask = mask if not inverted else 1.0 - mask
    return (base + mask).clamp(min=0.0, max=1.0)


def apply_mask(image: np.ndarray, mask: torch.Tensor, threshold: float, invert: bool, divide_by_255=True) -> np.ndarray:
    with torch.no_grad():
        channel_size = image.shape[-1]
        has_alpha_channel = channel_size == 4
        color = np.array(list(SELECTION_COLOR) + [255]) if has_alpha_channel else SELECTION_COLOR
        color = color.reshape((1, 1, -1))
        if divide_by_255:
            color = color / 255
        thresholded_mask = mask >= threshold if not invert else mask <= threshold
        mask = np.tile(torch_to_np((thresholded_mask))[..., None], (1, 1, channel_size))
        return np.where(mask, color, image)


def add_selection_masks(image: np.ndarray, state, use_saliency_mask, saliency_threshold, invert_saliency_mask,
                        use_depth_mask, depth_threshold, invert_depth_mask, divide_by_255=True):
    if use_saliency_mask:
        image = apply_mask(image, state.saliency_mask, saliency_threshold, invert_saliency_mask, divide_by_255)
    if use_depth_mask:
        image = apply_mask(image, state.depth_mask, depth_threshold, invert_depth_mask, divide_by_255)
    return image


def load_changes(state, file):
    editing_state = pickle.load(file)
    state.vp = editing_state['vp'].cuda()
    state.nst_image = editing_state['nst_image'].cuda()
    state.segmentation_labels = editing_state['segmentation_labels'].cuda()
    state.segmented_image = editing_state['segmented_image'].cuda()
    state.content_image = editing_state['content_image'].cuda()
    state.prev_nst_model_path = editing_state['prev_nst_model_path']
    state.nst_model = load_nst_model(state.prev_nst_model_path).cuda()


def save_changes(state, base_dir):
    editing_state = {'vp': state.vp.detach().cpu(),
                     'nst_image': state.nst_image.detach().cpu(),
                     'segmentation_labels': state.segmentation_labels.detach().cpu(),
                     'segmented_image': state.segmented_image.detach().cpu(),
                     'content_image': state.content_image.detach().cpu(),
                     'prev_nst_model_path': state.prev_nst_model_path}

    file_path = base_dir / 'editing_state.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(editing_state, file)
    st.info(f"changes written to {file_path}")


# Histogram matching with masked image
# Taken from: https://gist.github.com/tayden/dcc83424ce55bfb970f60db3d4ddad18
# Added a reference mask.

# image, reference [0-1] and image mask is a bool array.
def masked_match_histograms(image, image_mask, reference, reference_mask, fill_value):
    masked_image = np.ma.array(image, mask=image_mask)
    masked_reference = np.ma.array(reference, mask=reference_mask)
    matched = np.ma.array(np.empty(image.shape, dtype=image.dtype),
                          mask=image_mask, fill_value=fill_value)

    for channel in range(masked_image.shape[-1]):
        matched_channel = exposure.match_histograms(masked_image[..., channel].compressed(),
                                                    masked_reference[..., channel].compressed())

        # Re-insert masked background
        mask_ch = image_mask[..., channel]
        matched[..., channel][~mask_ch] = matched_channel.ravel()

    return matched.filled()
