import os
import sys
from pathlib import Path
from parameter_optimization.strotss_org import pil_resize_long_edge_to
from parameter_prediction_network.ast.ast_ppn_model import AST_PPN_MODEL

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from brushstroke.paint_transformer.inference.image_painter import BrushstrokeType, FirstStageImagePainter
from depth_estimation.DPT.run_monodepth import predict_depth
from effects import get_default_settings, ArbitraryStyleEffect
from effects.arbitrary_style_first_stage import perform_first_stage_segmentation, perform_first_stage_paint_transformer
from helpers import np_to_torch
from saliency_estimation.LDF.train_fine.test import predict_saliency
# from tools.utils import torch_to_display_image


@st.cache(hash_funcs={ArbitraryStyleEffect: id})
def create_effect(effect_type):
    effect, preset, param_set = get_default_settings(effect_type)
    effect.enable_checkpoints()
    effect.cuda()
    return effect, preset, param_set

def configure_effect_xdog(effect, effect_input, use_content_for_xdog):
    if use_content_for_xdog:
        effect.contours_on_content = True
        c = effect_input if st.session_state.get("stylized_image_editing", False) else st.session_state["Content_im"] 
        c = c if isinstance(c, torch.Tensor) else np_to_torch(c, add_batch_dim=True, divide_by_255=True).cuda()
        # resize c to effect_input size
        c = torch.nn.functional.interpolate(c, effect_input.shape[-2:], mode="bilinear")
        effect.content_img = c
    else:
        effect.contours_on_content = False
        effect.content_img = None

def load_visual_params(vp_path: str, img_org: Image, org_cuda: torch.Tensor, effect, preset) -> torch.Tensor:
    if Path(vp_path).exists():
        vp = torch.load(vp_path).detach().clone()
        vp = F.interpolate(vp, (img_org.height, img_org.width))
        if len(effect.vpd.vp_ranges) == vp.shape[1]:
            return vp
    # use preset and save it
    vp = effect.vpd.preset_tensor(preset, org_cuda, add_local_dims=True)
    torch.save(vp, vp_path)
    return vp


@st.experimental_memo
def load_params(content_id, style_id, effect_type, _effect, _preset):  # , effect):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    preoptim_param_path = os.path.join(dir_path, "precomputed", effect_type, content_id, style_id)
    img_path = os.path.join(preoptim_param_path, "input.png")
    # if not os.path.exists(img_path):
    #     img_path = os.path.join("tools/huggingface_demo", img_path)
    img_org = Image.open(img_path)
    content_cuda = np_to_torch(img_org).cuda()
    vp_path = os.path.join(preoptim_param_path, "vp.pt")
    # if not os.path.exists(vp_path):
    #     vp_path = os.path.join("tools/huggingface_demo", vp_path)
    vp = load_visual_params(vp_path, img_org, content_cuda, _effect, _preset)
    return content_cuda, vp


@st.cache(hash_funcs={FirstStageImagePainter: id})
def create_paint_transformer(brushstroke_type: str, skip_levels: int, num_detail_layers: int):
    if brushstroke_type == 'rectangle':
        return FirstStageImagePainter(BrushstrokeType.RECTANGLE, num_skip_last_drawing_layers=skip_levels,
                                      num_of_detail_layers=num_detail_layers).to('cuda')
    elif brushstroke_type == 'circle':
        return FirstStageImagePainter(BrushstrokeType.CIRCLE, num_skip_last_drawing_layers=skip_levels,
                                      num_of_detail_layers=num_detail_layers).to('cuda')
    elif brushstroke_type == 'brush':
        return FirstStageImagePainter(BrushstrokeType.BRUSH, num_skip_last_drawing_layers=skip_levels,
                                      num_of_detail_layers=num_detail_layers).to('cuda')
    else:
        raise Exception('unknown brushstroke')

@st.cache(hash_funcs={torch.Tensor: id})
def get_detail_mask(content_img, detail_mask_option, mask_needed): 
    if isinstance(content_img, Image.Image):
        content_img = np_to_torch(pil_resize_long_edge_to(content_img, 720)).cuda()
    if not mask_needed:
        return torch.ones_like(content_img)
    if detail_mask_option == 'depth mask':
        return predict_depth(content_img)
    elif detail_mask_option == 'saliency mask':
        return predict_saliency(content_img)
    else:
        return torch.ones_like(content_img)


def create_segmentation(effect_input_cuda, first_stage_config):
    if first_stage_config.first_stage == 'SLIC':
        _, _, segmented_image = perform_first_stage_segmentation(effect_input_cuda, n_segments=first_stage_config.num_segments)
    elif first_stage_config.first_stage == 'paint_transformer':
        paint_transformer = create_paint_transformer(first_stage_config.ptt_brush_stroke_type,
                                                     first_stage_config.skip_levels,
                                                     first_stage_config.num_detail_levels)
        content_for_detail_mask = effect_input_cuda if \
            st.session_state.get("stylized_image_editing", False) else st.session_state["Content_im"]
        detail_mask = get_detail_mask(content_for_detail_mask, first_stage_config.detail_mask_option,
                                      first_stage_config.num_detail_levels > 0)
        
        torch_to_display_image(detail_mask).save("detail_mask.jpg")

        _, segmented_image = perform_first_stage_paint_transformer(effect_input_cuda, paint_transformer, detail_mask, first_stage_config.prescale_factor)
    elif first_stage_config.first_stage == 'load':
        segmented_image = first_stage_config.custom_seg_image
    return segmented_image

@st.cache(hash_funcs={torch.Tensor: id})
def predict_depth_and_saliency(effect_input_cuda):
    content_for_detail_mask = effect_input_cuda if \
            st.session_state.get("stylized_image_editing", False) else st.session_state["Content_im"]
    depth_mask = get_detail_mask(content_for_detail_mask, "depth mask", True)
    saliency_mask = get_detail_mask(content_for_detail_mask, "saliency mask", True)
    return depth_mask, saliency_mask

@st.cache(hash_funcs={AST_PPN_MODEL: id})
def get_ast_model():
    ast_model = AST_PPN_MODEL.load_from_checkpoint("trained_models/AST/last.ckpt")
    ast_model.eval()
    ast_model = ast_model.cuda()
    return ast_model