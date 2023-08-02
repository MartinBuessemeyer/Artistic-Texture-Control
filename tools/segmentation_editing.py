from typing import Tuple

import numpy as np
import streamlit as st
import torch

from effects.arbitrary_style_first_stage import render_labels, segment_image
from helpers import torch_to_np, np_to_torch, save_as_image
from helpers.color_conversion import rgb_to_hsv, hsv_to_rgb
from tools.utils import SELECTION_COLOR, hex_color_to_int_tuple, \
    add_selection_masks, SEC_SELECTION_COLOR, masked_match_histograms

MERGE_BACKGROUND_SEGMENTS_EDITING_MODE = 'Merge Background Segments'
BACKGROUND_COLOR_PICKER_EDITING_MODE = 'Color Interpolation'
COPY_FORMAT_EDITING_MODE = 'Color Palette Matching'
COPY_STRUCTURE_EDITING_MODE = 'Copy Region'
RESEGMENTATION_EDITING_MODE = 'Change Level of Detail'
REPREDICT_PARAMETERS_EDITING_MODE = 'Re-predict Visual Parameters'

FREEDRAW_DRAWING_MODE = 'freedraw'

SEGMENT_EDITING_MODES = [COPY_FORMAT_EDITING_MODE, COPY_STRUCTURE_EDITING_MODE, RESEGMENTATION_EDITING_MODE,
                         REPREDICT_PARAMETERS_EDITING_MODE,
                         BACKGROUND_COLOR_PICKER_EDITING_MODE]

EMPTY_CANVAS_STATE_TEMPLATE = {'version': '4.4.0', 'objects': []}


def handle_background_segment_merge(canvas_image, state):
    with torch.no_grad():
        selected_segments, selected_segment_region = get_original_selected_segments_and_region(canvas_image, state)
        example_segment_id = selected_segments[0]
        state.segmentation_labels[selected_segment_region] = example_segment_id
        state.segmented_image = render_labels(state.segmentation_labels,
                                              state.nst_image.detach().clone(), 0, 0)


def get_original_selected_segments_and_region(canvas_image: np.ndarray, state, color: np.ndarray = SELECTION_COLOR,
                                              expand_dims=False) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    selected_image_region = get_selected_image_region(canvas_image, state, color)
    selected_segments = torch.unique(state.segmentation_labels[selected_image_region])
    selected_segment_region = torch.isin(state.segmentation_labels, selected_segments)
    if expand_dims:
        selected_segment_region = selected_segment_region.expand(-1, 3, -1, -1)
    return selected_segments, selected_segment_region


def get_selected_image_region(canvas_image, state, color: np.ndarray = SELECTION_COLOR):
    rgb_img = canvas_image[..., :3]
    alpha_img = canvas_image[..., 3]
    selected_image_region = (rgb_img == color).all(2) & alpha_img > 0
    selected_image_region = torch.tensor(selected_image_region,
                                         device=state.segmentation_labels.device).unsqueeze(0).unsqueeze(0)
    return selected_image_region


def handle_backgound_color_picked(canvas_image: np.ndarray, state, use_content_img_for_interpolation: bool,
                                  interpolation_value: float, segment_color: str, apply_edit):
    segmented_image, nst_image = interpolate_background_colors(state.segmented_image.detach().clone(),
                                                               canvas_image, state,
                                                               use_content_img_for_interpolation,
                                                               interpolation_value, segment_color)
    if apply_edit:
        state.segmented_image = segmented_image
        state.nst_image = nst_image
    return segmented_image


def interpolate_background_colors(target_image: torch.Tensor, canvas_image: np.ndarray, state,
                                  use_content_img_for_interpolation: bool, interpolation_value: float,
                                  segment_color: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        selected_segments, selected_segment_region = get_original_selected_segments_and_region(canvas_image, state)
        if len(selected_segments) == 0:
            return target_image, state.nst_image.detach().clone()
        segmentation_labels_subset = torch.where(selected_segment_region, state.segmentation_labels, 0)
        other_interpolation_candidate = state.content_image.detach().clone() if use_content_img_for_interpolation else \
            torch.tensor(hex_color_to_int_tuple(segment_color), device=target_image.device).view(-1, 3, 1, 1) / 255.0
        interpolated_image = torch.lerp(
            state.nst_image.detach().clone(),
            other_interpolation_candidate, interpolation_value)
        selected_segment_region = selected_segment_region.expand(-1, 3, -1, -1)
        adjusted_nst_image = state.nst_image.detach().clone()
        adjusted_nst_image[selected_segment_region] = interpolated_image[selected_segment_region]

        segmented_interpolated_image = render_labels(segmentation_labels_subset, interpolated_image,
                                                     0, 0)
        target_image[selected_segment_region] = segmented_interpolated_image[selected_segment_region]
        return target_image, adjusted_nst_image


def handle_copy_format(canvas_image: np.ndarray, state, apply: bool) -> torch.Tensor:
    transferred_image = copy_format(canvas_image, state.nst_image, state)
    resegmented_image = render_labels(state.segmentation_labels, transferred_image,
                                      sigma=0.0, kernel_size=0)
    if apply:
        state.nst_image = transferred_image
        state.segmented_image = resegmented_image
    return resegmented_image


def copy_format(canvas_image: np.ndarray, target_image: torch.Tensor, state) -> torch.Tensor:
    with torch.no_grad():
        dest_segments, dest_segment_region = get_original_selected_segments_and_region(canvas_image, state,
                                                                                       expand_dims=True)
        source_segments, source_segment_region = get_original_selected_segments_and_region(canvas_image, state,
                                                                                           expand_dims=True,
                                                                                           color=SEC_SELECTION_COLOR)
        if len(source_segments) == 0 or len(dest_segments) == 0:
            return target_image
        transfer_image = torch_to_np(rgb_to_hsv(target_image))
        matched_masked_image = masked_match_histograms(transfer_image, torch_to_np(~dest_segment_region),
                                                       transfer_image, torch_to_np(~source_segment_region),
                                                       0.0)
        matched_masked_image = hsv_to_rgb(np_to_torch(matched_masked_image).to(target_image.device))
        transferred_image = target_image.detach().clone()
        transferred_image[dest_segment_region] = matched_masked_image[dest_segment_region]
        return transferred_image


def get_bbox_of_selection(selected_region: torch.Tensor) -> Tuple[int, int, int, int]:
    x_idxs, y_idxs = torch.where(selected_region[0, 0])
    x_min, x_max = x_idxs.min(), x_idxs.max()
    y_min, y_max = y_idxs.min(), y_idxs.max()
    return x_min, x_max, y_min, y_max


def handle_resegmentation(canvas_image, state, num_segments_factor):
    with torch.no_grad():
        selected_segments, selected_segment_region = get_original_selected_segments_and_region(canvas_image, state)
        if len(selected_segments) == 0:
            return

        x_min, x_max, y_min, y_max = get_bbox_of_selection(selected_segment_region)
        sub_original_segmentation_labels = state.segmentation_labels[0, 0, x_min:x_max + 1,
                                           y_min:y_max + 1]
        sub_nst_image = \
            state.nst_image.detach().clone()[:, :, x_min:x_max + 1, y_min:y_max + 1]
        num_new_segments = int(len(selected_segments) * num_segments_factor) + 1
        sub_selected_segments_region = torch.isin(sub_original_segmentation_labels, selected_segments)
        sub_nst_image[~sub_selected_segments_region.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)] = 0
        sub_new_segmentation_labels = segment_image(sub_nst_image, num_new_segments)
        print(
            f'Resegmented {len(selected_segments)} segments to {len(sub_new_segmentation_labels.unique()) - 1} segments.')
        min_new_segmentation_idx = state.segmentation_labels.max() + 1
        sub_new_segmentation_labels += min_new_segmentation_idx
        sub_new_segmentation_labels = torch.where(sub_selected_segments_region, sub_new_segmentation_labels,
                                                  sub_original_segmentation_labels)
        state.segmentation_labels[:, :, x_min:x_max + 1, y_min:y_max + 1] = sub_new_segmentation_labels
        state.segmented_image = render_labels(state.segmentation_labels,
                                              state.nst_image.detach().clone(), sigma=0.0, kernel_size=0)


def handle_copy_structure(canvas_image: np.ndarray, state, apply: bool) -> torch.Tensor:
    copied_nst_image, copied_segmentation_labels, copied_segemented_image = copy_structure(canvas_image, state)
    if apply:
        state.nst_image = copied_nst_image
        state.segmentation_labels = copied_segmentation_labels
        state.segmented_image = copied_segemented_image
    return copied_segemented_image


def copy_structure(canvas_image: np.ndarray, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        dest_segments, dest_segment_region = get_original_selected_segments_and_region(canvas_image, state,
                                                                                       expand_dims=True)
        source_segments, source_segment_region = get_original_selected_segments_and_region(canvas_image, state,
                                                                                           expand_dims=True,
                                                                                           color=SEC_SELECTION_COLOR)
        if len(source_segments) == 0 or len(dest_segments) == 0:
            return state.nst_image.detach().clone(), state.segmentation_labels.detach().clone(), state.segmented_image.detach().clone()
        dest_x_min, dest_x_max, dest_y_min, dest_y_max = get_bbox_of_selection(dest_segment_region)
        x_dest_center = dest_x_min + (dest_x_max - dest_x_min) // 2
        y_dest_center = dest_y_min + (dest_y_max - dest_y_min) // 2

        src_x_min, src_x_max, src_y_min, src_y_max = get_bbox_of_selection(source_segment_region)
        x_src_center = src_x_min + (src_x_max - src_x_min) // 2
        y_src_center = src_y_min + (src_y_max - src_y_min) // 2

        x_shift = x_dest_center - x_src_center
        y_shift = y_dest_center - y_src_center

        shifted_x_min = src_x_min + x_shift
        shifted_x_max = src_x_max + x_shift
        shifted_y_min = src_y_min + y_shift
        shifted_y_max = src_y_max + y_shift

        # If the shift is outside the image, cut of the overlapping region.
        if shifted_x_min < 0:
            diff = abs(shifted_x_min)
            shifted_x_min = 0
            src_x_min += diff
        if shifted_y_min < 0:
            diff = abs(shifted_y_min)
            shifted_y_min = 0
            src_y_min += diff

        # Early out if the shift is so large, that no copied region remains.
        if src_y_min >= src_y_max or src_x_min >= src_x_max:
            return state.nst_image.detach().clone(), state.segmentation_labels.detach().clone(), state.segmented_image.detach().clone()

        shifted_source_region = torch.full_like(source_segment_region, False)
        shifted_source_region[..., shifted_x_min:shifted_x_max + 1,
        shifted_y_min:shifted_y_max + 1] = source_segment_region[
                                           ...,
                                           src_x_min:src_x_max + 1,
                                           src_y_min:src_y_max + 1]

        copied_nst_image = state.nst_image.detach().clone()
        copied_nst_image[shifted_source_region] = copied_nst_image[source_segment_region]
        copied_segment_labels = state.segmentation_labels.detach().clone()
        label_offset = torch.max(copied_segment_labels) + 1
        copied_segment_labels[shifted_source_region[0:1, 0:1, ...]] = copied_segment_labels[
                                                                          source_segment_region[0:1, 0:1,
                                                                          ...]] + label_offset

        copied_segmented_image = render_labels(copied_segment_labels, copied_nst_image,
                                               sigma=0.0, kernel_size=0)
        return copied_nst_image, copied_segment_labels, copied_segmented_image


def handle_repredict_parameters(state):
    if state.ppn_model is None:
        st.info('No Parameter Prediction Model is loaded. No Prediction Done.')
        return
    new_vp = state.ppn_model.predict_params(state.segmented_image)
    state.vp = torch.nn.functional.interpolate(new_vp, state.segmented_image.shape[-2:])


def perform_segmentation_editing(canvas_result, state, editing_mode, segment_color, num_segments_factor,
                                 saliency_threshold, depth_threshold, use_saliency_mask, use_depth_mask,
                                 invert_saliency_mask, invert_depth_mask,
                                 use_content_img_for_interpolation, interpolation_value,
                                 apply_edit):
    # Only perform editing action if canvas data is present.
    if canvas_result.image_data is None or canvas_result.json_data is None:
        return state.segmented_image.detach().clone()
    canvas_image_with_mask_selection = canvas_result.image_data
    canvas_image_with_mask_selection = add_selection_masks(canvas_image_with_mask_selection, state,
                                                           use_saliency_mask, saliency_threshold,
                                                           invert_saliency_mask,
                                                           use_depth_mask, depth_threshold, invert_depth_mask,
                                                           divide_by_255=False)

    _, selected_segment_region = get_original_selected_segments_and_region(
        canvas_image_with_mask_selection, state, expand_dims=True)
    selected_segment_region = selected_segment_region.float()
    _, selected_segment_region_black = get_original_selected_segments_and_region(
        canvas_image_with_mask_selection, state, color=SEC_SELECTION_COLOR, expand_dims=True)
    if torch.any(selected_segment_region_black):
        selected_segment_region_black = selected_segment_region_black.float()
        selected_segment_region[:, 0, ...] = selected_segment_region_black[:, 0, ...]
    save_as_image(selected_segment_region, 'tools/selection_mask.png')

    # Apply edits that have a preview.
    if editing_mode == BACKGROUND_COLOR_PICKER_EDITING_MODE:
        return handle_backgound_color_picked(canvas_image_with_mask_selection, state,
                                             use_content_img_for_interpolation,
                                             interpolation_value, segment_color, apply_edit)
    if editing_mode == COPY_FORMAT_EDITING_MODE:
        return handle_copy_format(canvas_image_with_mask_selection, state, apply_edit)
    if editing_mode == COPY_STRUCTURE_EDITING_MODE:
        return handle_copy_structure(canvas_image_with_mask_selection, state, apply_edit)

    # Apply edits that do not have a preview.
    if apply_edit:
        if editing_mode == MERGE_BACKGROUND_SEGMENTS_EDITING_MODE:
            handle_background_segment_merge(canvas_image_with_mask_selection, state)
        elif editing_mode == REPREDICT_PARAMETERS_EDITING_MODE:
            handle_repredict_parameters(state)
        elif editing_mode == RESEGMENTATION_EDITING_MODE:
            handle_resegmentation(canvas_image_with_mask_selection, state, num_segments_factor)

    return state.segmented_image.detach().clone()
