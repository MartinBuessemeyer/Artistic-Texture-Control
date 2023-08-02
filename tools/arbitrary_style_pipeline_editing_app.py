from copy import deepcopy
from pathlib import Path

import numpy as np
import streamlit as st
import torch.nn.functional as F
from matplotlib.colors import to_hex
from streamlit_drawable_canvas import st_canvas, _data_url_to_image

import helpers.session_state as session_state
from depth_estimation.DPT.run_monodepth import predict_depth
from effects import get_default_settings
from effects.arbitrary_style_first_stage import perform_first_stage_with_nst, apply_blur
from helpers import torch_to_np, load_image_cuda
from parameter_prediction_network.ppn_model import PPNModel
from saliency_estimation.LDF.train_fine.test import predict_saliency
from tools.custom_component.custom_js_and_css_component import add_custom_js_and_css
from tools.param_editing import get_edited_param, highlight_selected_segments, DRAWING_EDITING_MODE, \
    MASK_INTERPOLATION_EDITING_MODE, interpolate_param_with_masks
from tools.segmentation_editing import perform_segmentation_editing, RESEGMENTATION_EDITING_MODE, \
    FREEDRAW_DRAWING_MODE, EMPTY_CANVAS_STATE_TEMPLATE, BACKGROUND_COLOR_PICKER_EDITING_MODE, \
    COPY_FORMAT_EDITING_MODE, handle_repredict_parameters, SEGMENT_EDITING_MODES, COPY_STRUCTURE_EDITING_MODE
from tools.utils import load_nst_model, SELECTION_COLOR, \
    save_changes, torch_to_display_image, np_to_display_image, \
    add_selection_masks, load_changes, SEC_SELECTION_COLOR

# Specify canvas parameters in application
SEGMENTATION_EDITING_NAME = 'Segmentation Editing'
SEGMENT_SELECTION_CANVAS_KEY = 'canvas-segment-selection-key'
FREEDRAW_CANVAS_KEY = 'freedraw-selection-key'
BASE_DIR = Path('tools')

state = session_state.get(reruns_needed=0, canvas_key=SEGMENT_SELECTION_CANVAS_KEY,
                          content_image=None, nst_image=None, segmentation_labels=None, segmented_image=None,
                          nst_model=None, ppn_model=None, vp=None, depth_mask=None, saliency_mask=None,
                          prev_nst_model_path=None, prev_editing_state_path=None, prev_content_img_path=None,
                          prev_ppn_path=None)


def display_vp_and_mask_preview():
    vp_columns = st.columns(len(param_set) + 2)
    for i, p in enumerate(param_set):
        idx = effect.vpd.name2idx[p]
        vp_image = F.interpolate(state.vp[:, idx:idx + 1] + 0.5, (int(state.vp.shape[2] * 0.2),
                                                                  int(state.vp.shape[3] * 0.2)))
        vp_columns[i].text(p[:9])
        vp_columns[i].image(torch_to_np(vp_image), clamp=True)
    if state.depth_mask is not None and state.saliency_mask is not None:
        for idx, (name, img) in enumerate([('Depth Mask', state.depth_mask), ('Saliency Mask', state.saliency_mask)],
                                          1):
            mask_image = F.interpolate(img, (int(state.vp.shape[2] * 0.2),
                                             int(state.vp.shape[3] * 0.2)))
            vp_columns[-idx].text(name)
            vp_columns[-idx].image(torch_to_np(mask_image), clamp=True)


st.set_page_config(layout="wide")
add_custom_js_and_css()

loading_options_changed = False
loading_options = st.sidebar.expander('Loading Options')
im_path = loading_options.file_uploader("Load content image:", type=["png", "jpg"])
if not im_path:
    st.info("Please select an image first.")
    st.stop()

if state.prev_content_img_path is None or state.prev_content_img_path != im_path:
    state.content_image = load_image_cuda(im_path)
    state.prev_content_img_path = im_path

nst_model_path = loading_options.file_uploader("Load Johnson NST checkpoint:", type=["ckpt"])
if not nst_model_path:
    st.info("Please select a Johnson NST checkpoint first.")
    st.stop()

if state.nst_model is None or state.prev_nst_model_path != nst_model_path:
    state.nst_model = load_nst_model(nst_model_path).cuda()
    state.prev_nst_model_path = nst_model_path
    loading_options_changed = True

effect, preset, param_set = get_default_settings('arbitrary_style')
effect = effect.cuda()

vp_path = loading_options.file_uploader("Load visual parameters:", type=["pt"])

if state.vp is None:
    state.vp = effect.vpd.preset_tensor(preset, state.content_image, add_local_dims=True)

editing_state_path = loading_options.file_uploader("Load Editing State:", type=["pickle"])

if state.prev_editing_state_path != editing_state_path:
    if editing_state_path is not None:
        load_changes(state, editing_state_path)
    state.prev_editing_state_path = editing_state_path

ppn_path = loading_options.file_uploader("Load PPN Model", type=['ckpt'])

effect_and_vps_match = len(param_set) == state.vp.shape[1]

if not effect_and_vps_match:
    st.error(
        f'Currently have {state.vp.shape[1]} different visual parameters but effect arbitrary style expects {len(param_set)}')

if state.nst_image is None or loading_options_changed:
    state.nst_image, state.segmentation_labels, state.segmented_image = \
        perform_first_stage_with_nst(state.content_image, state.nst_model, 0, 0.0)
    state.depth_mask = predict_depth(state.content_image)
    state.saliency_mask = predict_saliency(state.content_image)

if ppn_path is not None and ppn_path != state.prev_ppn_path:
    state.prev_ppn_path = ppn_path
    state.ppn_model = PPNModel.load_from_checkpoint(ppn_path, strict=False).cuda()
    handle_repredict_parameters(state)

smoothing_options = st.sidebar.expander('Smoothing Options')
sigma = smoothing_options.slider("Sigma: ", 0.0, 10.0, 1.0, 0.05)
kernel_size = smoothing_options.slider("Kernel Size: ", 1, 9, 3, 2)

mask_options = st.sidebar.expander('Property Masks')
use_saliency_mask = mask_options.checkbox('Use Saliency Mask', False)
invert_saliency_mask = mask_options.checkbox('Invert Saliency Mask', False)
saliency_threshold = mask_options.slider("Saliency Threshold: ", 0.0, 1.0, 1.0, 0.05)
use_depth_mask = mask_options.checkbox('Use Depth Mask', False)
invert_depth_mask = mask_options.checkbox('Invert Depth Mask', False)
depth_threshold = mask_options.slider("Depth Threshold: ", 0.0, 1.0, 1.0, 0.05)

active_param = st.sidebar.selectbox("Active Parameter: ", [SEGMENTATION_EDITING_NAME] + param_set)

coll1, coll2, coll3 = st.columns([1, 1, 1])


def render_param_mask_editing(state):
    st.sidebar.text("Drawing options")
    edit_mode = st.sidebar.selectbox('Edit Mode:', [DRAWING_EDITING_MODE, MASK_INTERPOLATION_EDITING_MODE])
    interpolation_strength = 0.0
    stroke_width = 0
    if edit_mode == DRAWING_EDITING_MODE:
        # Normal parameter editing stuff
        absolute_change = st.sidebar.checkbox("Change Value absolute")
        edit_on_borders = st.sidebar.checkbox('Perform Edit on Border of marked areas')
        border_size = 1
        if edit_on_borders:
            border_size = st.sidebar.slider('Border Size', 1, 10, 1, 1)
        value_change = st.sidebar.slider("Param Value Change: ", -1.0, 1.0, 0.0, 0.01)
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3)
    else:
        interpolation_strength = st.sidebar.slider("Interpolation Strength:", 0.0, 1.0, 0.1, 0.01)
    perform_param_edit = st.sidebar.button('Perform Param Edit')
    reset_selection = st.sidebar.button('Reset All Region Selection')
    stroke_color = to_hex(SELECTION_COLOR / 255.0)
    coll1.header("Segment Selection")
    coll2.header("Draw Mask")
    coll3.header("Live Output")
    if SEGMENT_SELECTION_CANVAS_KEY in st.session_state.keys() and \
            'data' in st.session_state[SEGMENT_SELECTION_CANVAS_KEY].keys():
        canvas_img = np.asarray(_data_url_to_image(st.session_state[SEGMENT_SELECTION_CANVAS_KEY]['data']))
        segmentation_background = highlight_selected_segments(canvas_img, state)
    else:
        segmentation_background = state.segmented_image.detach().clone()
    segmentation_background_img = torch_to_np(segmentation_background)
    if edit_mode == DRAWING_EDITING_MODE:
        segmentation_background_img = add_selection_masks(segmentation_background_img, state,
                                                          use_saliency_mask, saliency_threshold,
                                                          invert_saliency_mask,
                                                          use_depth_mask, depth_threshold,
                                                          invert_depth_mask)
    segmentation_background_img = np_to_display_image(segmentation_background_img)
    with coll1:
        segmentation_selection_canvas_result = st_canvas(
            fill_color=stroke_color,  # Fixed fill color with some opacity
            background_color="rgb(1, 1, 1)",
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            background_image=segmentation_background_img,
            update_streamlit=True,
            width=segmentation_background_img.width,
            height=segmentation_background_img.height,
            drawing_mode=FREEDRAW_DRAWING_MODE,
            key=SEGMENT_SELECTION_CANVAS_KEY,
            initial_drawing=deepcopy(EMPTY_CANVAS_STATE_TEMPLATE) if reset_selection or perform_param_edit else None,
            display_toolbar=True
        )
    param_idx = effect.vpd.name2idx[active_param]
    with coll2:
        current_param = state.vp[:, param_idx:param_idx + 1]
        if edit_mode == MASK_INTERPOLATION_EDITING_MODE:
            current_param = interpolate_param_with_masks(current_param, use_saliency_mask, use_depth_mask,
                                                         invert_saliency_mask, invert_depth_mask,
                                                         interpolation_strength, state)
        current_param_img = np.expand_dims(torch_to_np(current_param + 0.5), axis=-1)
        if edit_mode == DRAWING_EDITING_MODE:
            current_param_img = add_selection_masks(current_param_img, state,
                                                    use_saliency_mask, saliency_threshold, invert_saliency_mask,
                                                    use_depth_mask, depth_threshold, invert_depth_mask)
        current_param_img = np_to_display_image(current_param_img)
        freedraw_canvas_result = st_canvas(
            fill_color=stroke_color,  # Fixed fill color with some opacity
            background_color="rgb(1, 1, 1)",
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            background_image=current_param_img,
            update_streamlit=True,
            width=current_param_img.width,
            height=current_param_img.height,
            drawing_mode=FREEDRAW_DRAWING_MODE,
            key=FREEDRAW_CANVAS_KEY,
            initial_drawing=deepcopy(EMPTY_CANVAS_STATE_TEMPLATE) if reset_selection or perform_param_edit else None,
            display_toolbar=True
        )
    edited_vp = state.vp.detach().clone()
    if segmentation_selection_canvas_result.image_data is not None and freedraw_canvas_result.image_data is not None:
        if edit_mode == DRAWING_EDITING_MODE:
            masked_canvas_result = add_selection_masks(freedraw_canvas_result.image_data, state,
                                                       use_saliency_mask, saliency_threshold, invert_saliency_mask,
                                                       use_depth_mask, depth_threshold, invert_depth_mask,
                                                       divide_by_255=False)
            edited_vp[:, param_idx:param_idx + 1] = get_edited_param(segmentation_selection_canvas_result.image_data,
                                                                     masked_canvas_result,
                                                                     edited_vp[:, param_idx:param_idx + 1],
                                                                     absolute_change, value_change,
                                                                     edit_on_borders, border_size, state)
        elif edit_mode == MASK_INTERPOLATION_EDITING_MODE:
            edited_vp[:, param_idx:param_idx + 1] = current_param

        if perform_param_edit:
            state.vp = edited_vp
        if reset_selection or perform_param_edit:
            state.reruns_needed = 1
    final_result = effect(state.segmented_image, edited_vp)
    with coll3:
        coll3.image(torch_to_display_image(final_result),
                    use_column_width=False)


def render_segment_editing(state):
    # Segmentation editing stuff
    editing_mode = st.sidebar.selectbox(
        "Editing tool:", (editing_mode for editing_mode in SEGMENT_EDITING_MODES), key='edit_select_box'
    )
    # color picking mode
    segment_color = None
    use_content_img_for_interpolation = True
    interpolation_value = 0.0
    if editing_mode == BACKGROUND_COLOR_PICKER_EDITING_MODE:
        use_content_img_for_interpolation = st.sidebar.checkbox('Use content image for color interpolation', value=True)
        other_interpolation_name = 'Content' if use_content_img_for_interpolation else 'Static Color'
        interpolation_value = st.sidebar.slider(f"Color interpolation: NST <-> {other_interpolation_name}",
                                                0.0, 1.0, 0.0, 0.05)
        if not use_content_img_for_interpolation:
            segment_color = st.sidebar.color_picker('Choose static interpolation color.')
    # copy format mode
    use_sec_color = False
    if editing_mode in [COPY_FORMAT_EDITING_MODE, COPY_STRUCTURE_EDITING_MODE]:
        use_sec_color = st.sidebar.checkbox('Mark Region to Copy', value=False)
    # resegment mode
    num_segments_factor = 1.0
    if editing_mode == RESEGMENTATION_EDITING_MODE:
        num_segments_factor = st.sidebar.slider("Number of Segments Factor: ", 0.00, 20.0, 1.0, 0.1,
                                                key='segment_factor_slider')
    # general purpose
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3, key='stroke_slicer')
    apply_edit = st.sidebar.button('Apply Edit', key='apply_button')
    coll1.header("Draw Mask")
    coll2.header("First Stage Result")
    coll3.header("Live Output")
    if st.session_state is not None and SEGMENT_SELECTION_CANVAS_KEY in st.session_state.keys():
        canvas_img = np.asarray(_data_url_to_image(st.session_state[SEGMENT_SELECTION_CANVAS_KEY]['data']))
        drawing_background_img = torch_to_np(highlight_selected_segments(canvas_img, state))
    else:
        drawing_background_img = torch_to_np(state.segmented_image.detach().clone())
    drawing_background_img = add_selection_masks(drawing_background_img, state, use_saliency_mask, saliency_threshold,
                                                 invert_saliency_mask, use_depth_mask, depth_threshold,
                                                 invert_depth_mask)
    drawing_background_img = np_to_display_image(drawing_background_img)
    stroke_color = SELECTION_COLOR if not use_sec_color else SEC_SELECTION_COLOR
    stroke_color = to_hex(stroke_color / 255.0)
    with coll1:
        canvas_result = st_canvas(
            fill_color=stroke_color,  # Fixed fill color with some opacity
            background_color="rgb(1, 1, 1)",
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            background_image=drawing_background_img,
            update_streamlit=True,
            width=drawing_background_img.width,
            height=drawing_background_img.height,
            drawing_mode=FREEDRAW_DRAWING_MODE,
            key=SEGMENT_SELECTION_CANVAS_KEY,
            initial_drawing=None,
            display_toolbar=True
        )
    edited_first_stage_output = perform_segmentation_editing(canvas_result, state, editing_mode, segment_color,
                                                             num_segments_factor,
                                                             saliency_threshold, depth_threshold, use_saliency_mask,
                                                             use_depth_mask,
                                                             invert_saliency_mask, invert_depth_mask,
                                                             use_content_img_for_interpolation, interpolation_value,
                                                             apply_edit)
    coll2.image(torch_to_display_image(edited_first_stage_output),
                use_column_width=False)
    final_result = apply_blur(edited_first_stage_output, kernel_size, sigma)
    final_result = effect(final_result, state.vp)
    coll3.image(torch_to_display_image(final_result),
                use_column_width=False)


if active_param != SEGMENTATION_EDITING_NAME:
    render_param_mask_editing(state)
else:
    render_segment_editing(state)

display_vp_and_mask_preview()

st.sidebar.text("Update")
save_changes_pressed = st.sidebar.button("Write changes", key='save_button')

if save_changes_pressed:
    save_changes(state, BASE_DIR)
