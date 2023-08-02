import numpy as np
import torch

from helpers.visual_parameter_def import VisualParameterDef
from tools.segmentation_editing import get_original_selected_segments_and_region, get_selected_image_region
from tools.utils import add_interpolation_mask, SELECTION_COLOR, SEC_SELECTION_COLOR

MASK_INTERPOLATION_EDITING_MODE = 'Interpolate Masks with Parameter'
DRAWING_EDITING_MODE = 'Change Parameters'


def get_edited_param(segmentation_canvas_img: np.ndarray, freedraw_canvas_img: np.ndarray,
                     param: torch.Tensor, absolute_change: bool, value_delta: float,
                     edit_on_borders: bool, border_size: int, state) -> torch.Tensor:
    with torch.no_grad():
        _, selected_segment_region = get_original_selected_segments_and_region(segmentation_canvas_img, state)
        selected_freedraw_region = get_selected_image_region(freedraw_canvas_img, state)
        selected_region = torch.logical_or(selected_segment_region, selected_freedraw_region)

        if edit_on_borders:
            # Get border by applying dilation (max pool) and then take the difference
            kernel_size = border_size * 2 + 1
            selected_region_with_borders = torch.nn.functional.max_pool2d(selected_region.float(), kernel_size,
                                                                          stride=1, padding=border_size).bool()
            selected_region = selected_region_with_borders & (~selected_region)

        if absolute_change:
            param[selected_region] = value_delta
        else:
            param[selected_region] += value_delta
        return VisualParameterDef.clamp_range(param)


def interpolate_param_with_masks(param: torch.Tensor, use_saliency_mask: bool, use_depth_mask: bool,
                                 invert_saliency_mask: bool, invert_depth_mask: bool,
                                 strength: float, state) -> torch.Tensor:
    with torch.no_grad():
        combined_masks = torch.zeros_like(param)
        if use_saliency_mask:
            combined_masks = add_interpolation_mask(state.saliency_mask, invert_saliency_mask, combined_masks)
        if use_depth_mask:
            combined_masks = add_interpolation_mask(state.depth_mask, invert_depth_mask, combined_masks)
        combined_masks = combined_masks - 0.5
        return torch.lerp(param, combined_masks, strength).clamp(-0.5, 0.5)


def interpolate_between_params_with_masks(param1: torch.Tensor, param2: torch.Tensor, use_saliency_mask: bool,
                                          use_depth_mask: bool,
                                          invert_saliency_mask: bool, invert_depth_mask: bool,
                                          strength: float, state) -> torch.Tensor:
    with torch.no_grad():
        combined_masks = torch.zeros_like(param1)
        if use_saliency_mask:
            combined_masks = add_interpolation_mask(state.saliency_mask, invert_saliency_mask, combined_masks)
        if use_depth_mask:
            combined_masks = add_interpolation_mask(state.depth_mask, invert_depth_mask, combined_masks)
        print("avg mask", torch.mean(combined_masks))
        return torch.lerp(param1, param2, combined_masks * strength).clamp(-0.5, 0.5)


def highlight_selected_segments(segmentation_canvas_img: np.ndarray, state) -> torch.Tensor:
    highlighted_segmentation = state.segmented_image.detach().clone()

    _, selected_segment_region = get_original_selected_segments_and_region(segmentation_canvas_img, state)
    selected_segment_region = selected_segment_region.expand((-1, 3, -1, -1))
    highlighted_segmentation[selected_segment_region] = SELECTION_COLOR[0] / 255.0

    _, selected_segment_region = get_original_selected_segments_and_region(segmentation_canvas_img, state,
                                                                           SEC_SELECTION_COLOR)
    selected_segment_region = selected_segment_region.expand((-1, 3, -1, -1))
    highlighted_segmentation[selected_segment_region] = SEC_SELECTION_COLOR[0] / 255.0
    return highlighted_segmentation
