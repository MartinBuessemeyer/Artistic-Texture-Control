import base64
import os
import sys
import time
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from streamlit.logger import get_logger


PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import helpers.session_state as session_state
from parameter_optimization.strotss_org import np_to_pil
from effect_helpers import configure_effect_xdog, create_effect, create_segmentation, get_ast_model, load_params, predict_depth_and_saliency
from effects import ArbitraryStyleEffect
from gui_helpers import clickable_image_row, get_image_urls, last_image_clicked, retrieve_from_id
from helpers import np_to_torch, pil_resize_long_edge_to, torch_to_np
from tasks import monitor_task, optimize, optimize_next, optimize_params
from parameter_prediction_network.ast.ast_ppn_model import AST_PPN_MODEL
import tools
from tools.param_editing import interpolate_between_params_with_masks, interpolate_param_with_masks

st.set_page_config(layout="wide")
BASE_URL = "https://ivpg.hpi3d.de/wise/wise-demo/images/"
LOGGER = get_logger(__name__)

effect_type = "arbitrary_style"

# if sys.version_info[0:2] not in [(3, 8), (3, 9)] :
#     raise Exception('Requires python 3.8 or 3.9') # TODO: check why it may not work with 3.10+

if "click_counter" not in st.session_state:
    st.session_state.click_counter = 1

if "action" not in st.session_state:
    st.session_state["action"] = ""

if "user" not in st.session_state:
    st.session_state["user"] = hash(time.time())

content_urls = [
    {"id": "portrait", "src": BASE_URL + "/content/portrait.jpeg"},
    {"id": "tubingen", "src": BASE_URL + "/content/tubingen.jpeg"},
    {"id": "colibri", "src": BASE_URL + "/content/colibri.jpeg"}
]

style_urls = [
    {"id": "starry_night", "src": BASE_URL + "/style/starry_night.jpg"},
    {"id": "the_scream", "src": BASE_URL + "/style/the_scream.jpg"},
    {"id": "wave", "src": BASE_URL + "/style/wave.jpg"},
    {"id": "woman_with_hat", "src": BASE_URL + "/style/woman_with_hat.jpg"}
]


def store_img_from_id(clicked, urls, imgtype):
    img, src = retrieve_from_id(clicked, urls)
    session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_render_src": src, f"{imgtype}_id": clicked})


def img_choice_panel(imgtype, urls, default_choice, expanded):
    with st.expander(f"Select {imgtype} image:", expanded=expanded):
        clicked = clickable_image_row(urls)

        non_click_states = ("uploaded", "switch_page_from_local_edits", 
            "switch_page_from_presets", "slider_change","reset")
        if not clicked and st.session_state["action"] not in non_click_states:  # default val
            store_img_from_id(default_choice, urls, imgtype)

        st.write("OR:  ")
        with st.form(imgtype + "-form", clear_on_submit=True):
            uploaded_im = st.file_uploader(f"Load {imgtype} image:", type=["png", "jpg", "jpeg"], )
            use_content_as_style = st.checkbox("Use only content image (=Stylized image editing)", 
                 value=False) if imgtype == "Content" else False

            upload_pressed = st.form_submit_button("Upload")
            if upload_pressed and uploaded_im is not None:
                img = Image.open(uploaded_im).convert('RGB')
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                src_decoded = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_id": "uploaded",
                    f"{imgtype}_render_src": src_decoded})
                if use_content_as_style:
                    st.session_state["stylized_image_editing"] = True
                    for key in ["Style_im", "Style_id", "Style_render_src"]:
                        st.session_state[key] = st.session_state[key.replace("Style", "Content")]
                else:
                    st.session_state["stylized_image_editing"] = False
                st.session_state["action"] = "uploaded"
                st.write("uploaded.")

        last_clicked = last_image_clicked(type=imgtype)
        print(st.session_state["user"], " last_clicked", last_clicked, "clicked", clicked, "action",
              st.session_state["action"])
        if not upload_pressed and clicked != "":  # trigger when no file uploaded
            if last_clicked != clicked:  # only activate when content was actually clicked
                store_img_from_id(clicked, urls, imgtype)
                last_image_clicked(type=imgtype, action=clicked)
                st.session_state["stylized_image_editing"] = False
                if "uploaded" in (st.session_state["Style_id"], st.session_state["Content_id"]):
                    st.session_state["action"] = "uploaded"
                else:
                    st.session_state["action"] = "clicked"
                st.session_state.click_counter += 1  # hack to get page to reload at top

        state = session_state.get()
        st.sidebar.write(f'Selected {imgtype} image:')
        st.sidebar.markdown(f'<img src="{state[f"{imgtype}_render_src"]}" width=240px></img>', unsafe_allow_html=True)


result_container = st.container()
coll1, coll2 = result_container.columns([3, 2])
coll1.header("Result")
coll2.header("Global Edits")
result_image_placeholder = coll1.empty()
if "last_result" not in st.session_state:
    result_image_placeholder.markdown("## loading..")
else:
    result_image_placeholder.image(st.session_state["last_result"], caption="COMPUTING...." )

if "current_server_task_id" not in st.session_state:
    st.session_state['current_server_task_id'] = None

if "optimize_next" not in st.session_state:
    st.session_state['optimize_next'] = False


st.session_state["effect_type"] = effect_type
load_from_dir = st.sidebar.checkbox("Load from directory", value=False)
if load_from_dir:
    content_urls, style_urls = get_image_urls("arbitrary_style_brushstroke")

# if st.session_state["action"] == "change_effect":
#     st.session_state["action"] = "reset"
#     st.experimental_rerun()

effect, preset, param_set = create_effect(effect_type)#st.session_state["effect_type"])

if st.session_state["optimize_next"]:
    print("optimize now")
    st.session_state["optimize_next"] = False
    # optimize(effect, preset, st.session_state["Style_im"], result_image_placeholder)

    style = st.session_state["Style_im"]
    with st.spinner(text="Optimizing parameters.."):
        print("optimizing for user", st.session_state["user"])
        optimize_params(effect, preset, st.session_state["Content_im"], style, result_image_placeholder, 
                        stylized_image_editing=st.session_state.get("stylized_image_editing", False))

    st.session_state["original_vp"] = st.session_state["result_vp"]


img_choice_panel("Content", content_urls, "portrait", expanded=True)
img_choice_panel("Style", style_urls, "starry_night", expanded=True)

state = session_state.get()
content_id = state["Content_id"]
style_id = state["Style_id"]

print("content id, style id", content_id, style_id)
if st.session_state["action"] == "uploaded":
    effect_input, _vp = optimize_next(result_image_placeholder)
elif st.session_state["action"] in ("switch_page_from_local_edits", "switch_page_from_presets", "slider_change") or \
        content_id == "uploaded" or style_id == "uploaded":
    print(st.session_state["user"], "restore param")
    _vp = st.session_state["result_vp"]
    effect_input = st.session_state["effect_input"]
else:
    print(st.session_state["user"], "load_params")
    effect_input, _vp = load_params(content_id, style_id, st.session_state["effect_type"], effect, preset)
    st.session_state["original_vp"] = _vp.clone()

vp = torch.clone(_vp)

def reset_params(means, names):
    for i, name in enumerate(names):
        st.session_state["slider_" + name] = means[i]


def on_slider():
    st.session_state["action"] = "slider_change"


def render_effect(effect, effect_input_cuda, vp, first_stage_config, pre_segmented_input=None):
    if isinstance(effect, ArbitraryStyleEffect):
        if pre_segmented_input is not None:
            effect_input_cuda = pre_segmented_input
        else:
            effect_input_cuda = create_segmentation(effect_input_cuda, first_stage_config)
        st.session_state["first_stage_output"] = effect_input_cuda
        if first_stage_config.show_first_stage_only:
            return Image.fromarray((torch_to_np(effect_input_cuda) * 255.0).astype(np.uint8))

    with torch.no_grad():
        result_cuda = effect(effect_input_cuda, vp)
    img_res = Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8))
    return img_res


with coll2:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["global params", "segmentation", "re-optimization",  "re-prediction", "interpolation", "config"])
    with tab1:
        show_params_names = ['bumpiness', "bumpSpecular", "contours"]
        display_means = []
        show_params_names = ['bump_opacity', "bump_scale", "bumpSpecular", "contour_opacity"]
        params_mapping = {"bump_opacity": ["bump_opacity"], "bump_scale": ["bump_scale"],
                            "bumpSpecular": ["bump_phong_specular", "bump_phong_shininess"],
                            "contour_opacity": ["contour_opacity"]}


        def create_slider(name):
            params = params_mapping[name] if name in params_mapping else [name]
            means = [torch.mean(vp[:, effect.vpd.name2idx[n]]).item() for n in params]
            display_mean = np.average(means) + 0.5
            display_means.append(display_mean)
            if "slider_" + name not in st.session_state or st.session_state["action"] != "slider_change":
                st.session_state["slider_" + name] = display_mean.astype(np.float32).item()
            slider = st.slider(f"Mean {name}: ", 0.0, 1.0, step=0.01, key="slider_" + name, on_change=on_slider)
            for i, param_name in enumerate(params):
                vp[:, effect.vpd.name2idx[param_name]] += slider - display_mean
            vp.clamp_(-0.5, 0.5)


        for name in show_params_names:
            create_slider(name)

        others_idx = set(range(len(effect.vpd.vp_ranges))) - set(
            [effect.vpd.name2idx[name] for name in sum(params_mapping.values(), [])])
        others_names = [effect.vpd.vp_ranges[i][0] for i in sorted(list(others_idx))]
        # if effect_selection == "minimal_pipeline":
        #     others_names = ["hueShift"] + [n for n in others_names if n != "hueShift"]
        other_param = st.selectbox("Other parameters: ", others_names)
        create_slider(other_param)

        reset_button = st.button("Reset Parameters", on_click=reset_params, args=(display_means, show_params_names))
        if reset_button:
            st.session_state["action"] = "reset"
            st.experimental_rerun()

    with tab2:
        num_segments = None; skip_levels = 0; segmentation_type = None; brushstroke_type = ""; show_first_stage_only=False
        detail_mask_option = ""; custom_seg_image=None; num_detail_levels = 0; paint_transformer_prescale_factor=1.0;
        # if effect_selection == "arbitrary_style_pipeline":
        segmentation_type = st.selectbox("Segmentation Type: ", ["SLIC", "paint_transformer", "load"], index=0, on_change=on_slider)
        if segmentation_type == "SLIC":
            num_segments = st.slider(f"Number of segments: ", 100, 5000, value=5000, step=10, on_change=on_slider)
            st.session_state["num_segments"] = num_segments
        elif segmentation_type == "paint_transformer":
            brushstroke_type = st.selectbox("brushstroke type:", ["rectangle", "circle", "brush"],
                                            on_change=on_slider)
            skip_levels = st.slider("skip levels:", 0, 4, value=0, step=1, on_change=on_slider)
            detail_mask_option = st.selectbox("Detail Mask Generator:", ["depth mask", "saliency mask", ""],
                                                on_change=on_slider)
            paint_transformer_prescale_factor = st.slider("prescale before painting by factor:", 1.0, 2.0, value=1.3, step=0.1, on_change=on_slider)
            num_detail_levels = st.slider("Number of detail levels:", 0, 5, value=0, step=1, on_change=on_slider) if detail_mask_option != "" else 0
        else:
            custom_seg_image = st.file_uploader("upload segmented image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
            if not custom_seg_image:
                custom_seg_image = st.session_state["effect_input"] if "custom_seg_image" not in st.session_state else st.session_state["first_stage_output"]
                st.warning("upload a segmentation image")
            else:
                custom_seg_image = torch.from_numpy(np.array(
                    Image.open(custom_seg_image).convert("RGB")).transpose(2,0,1)).cuda().unsqueeze(0) / 255.0
                custom_seg_image = torch.nn.functional.interpolate(custom_seg_image, size=effect_input.shape[-2:], mode="bilinear", align_corners=False)
                st.session_state["first_stage_output"] = custom_seg_image

        show_first_stage_only = st.checkbox("show first stage only", value=False, on_change=on_slider)

        firststage_config = SimpleNamespace(num_segments=num_segments, first_stage=segmentation_type,
                                            show_first_stage_only=show_first_stage_only,
                                            ptt_brush_stroke_type=brushstroke_type, skip_levels=skip_levels, custom_seg_image=custom_seg_image,
                                            num_detail_levels=num_detail_levels, detail_mask_option=detail_mask_option, 
                                            prescale_factor=paint_transformer_prescale_factor)
    with tab3:
        if isinstance(effect, ArbitraryStyleEffect):
            st.write('re-optimization:')
            optimize_from_scratch = st.checkbox("optimize params from scratch", value=True, on_change=on_slider)
            reoptimize_loss = st.selectbox("reoptimize loss:", ["L1", "STROTTS", "GatysLoss", "CLIPStyler"], index=0,
                                           on_change=on_slider)
            extra_kw_args = {}
            extra_kw_args["firststage_config"] = firststage_config
            if "stylized_image_editing" in st.session_state:
                extra_kw_args["stylized_image_editing"] = st.session_state["stylized_image_editing"]

            if reoptimize_loss == "STROTTS" or reoptimize_loss == "GatysLoss":
                # slider for content loss
                content_loss_weight = st.slider("content loss weight", 0.0, 100.0, value=16.0, step=1.0,
                                                on_change=on_slider)
                extra_kw_args["nst_weight_content"] = content_loss_weight
                segmented_target = st.checkbox("use segmented image as content target", value=False,
                                               help="Can be used to make output more abstract", on_change=on_slider)
                extra_kw_args["segmented_target"] = segmented_target
                different_style_image = st.checkbox("use different style image for parameter optim than original",
                                                    value=False, on_change=on_slider)
                if different_style_image:
                    style_image = st.file_uploader("upload style image", type=["png", "jpg", "jpeg"],
                                                   accept_multiple_files=False)
                    st.write("or")
                    selected_style_name = st.selectbox("select style", [f["id"] for f in style_urls])
                    style_image = retrieve_from_id(selected_style_name, style_urls)[0] \
                        if style_image is None else Image.open(style_image).convert("RGB")
                    extra_kw_args["different_style_image"] = style_image
            elif reoptimize_loss == "CLIPStyler":
                # add text field for clipstyler text prompt
                text_prompt = st.text_input("text prompt", value="sketch with a black pencil", on_change=on_slider)
                extra_kw_args["clipstyler_text_prompt"] = text_prompt
                use_content_loss = st.checkbox("use content loss", value=True, on_change=on_slider)
                extra_kw_args["clipstyler_use_content_loss"] = use_content_loss
                crop_size = st.slider("crop size", 0, 512, value=128, step=1, on_change=on_slider)
                extra_kw_args["clipstyler_crop_size"] = crop_size
            else: # l1
                segmented_target = False
            
            optim_steps = st.slider("optimization steps", 1, 200, value=100, step=1, on_change=on_slider)
            extra_kw_args["optim_steps"] = optim_steps
            use_brushstroke_loss = st.checkbox("use brushstroke loss", value=False, on_change=on_slider)
            extra_kw_args["use_brushstrokeloss"] = use_brushstroke_loss
            
            re_optimized_btn = st.button("Re-Optimize Parameters")
            if re_optimized_btn:
                optimize(effect, preset, st.session_state["Style_im"], result_image_placeholder,
                         start_from_preset=optimize_from_scratch, loss=reoptimize_loss, **extra_kw_args)
                vp = st.session_state["result_vp"]
    with tab4:
        use_different_style_image = st.checkbox("use different style image for parameter prediction")
        style_image = st.file_uploader("upload style image for AST", type=["png", "jpg", "jpeg"], accept_multiple_files=False) \
                if use_different_style_image else st.session_state["Style_im"]
        prediction_config = None
        repredict_btn = st.button("Re-Predict Parameters using arbitrary style transfer")
        pre_segmented_input = None
        if repredict_btn:
            if not style_image:
                st.warning("upload a style image")
            else:
                pre_segmented_input = create_segmentation(effect_input, firststage_config)
                style_image =  Image.open(style_image).convert("RGB") if not isinstance(style_image, Image.Image) else style_image
                style_image = np_to_torch(style_image, add_batch_dim=True, divide_by_255=True).cuda()
                # resize vp to nearest multiple of 32
                inp = torch.nn.functional.interpolate(effect_input, (effect_input.shape[-2] // 32 * 32, 
                    effect_input.shape[-1] // 32 * 32), mode="bilinear")
                _vp, _ = get_ast_model().predict_params(inp, style_image)
                vp = torch.nn.functional.interpolate(_vp, vp.shape[-2:], mode="bilinear")
                st.session_state["action"] = "slider_change"
    with tab5:
        # choose depth, saliency or none (default)
        depth_or_saliency = st.selectbox("interpolation", ["none", "depth", "saliency"], index=0, on_change=on_slider)
        mask_black_level = st.slider("mask black level", -1.0, 2.0, value=0.0, step=0.01, on_change=on_slider)
        mask_white_level = st.slider("mask white level", -1.0, 2.0, value=1.0, step=0.01, on_change=on_slider)

        interpolate_with_preset = st.selectbox("interpolate with preset", ["original_vp"] if "presets" not in st.session_state else 
                                               list(st.session_state["presets"].keys()), index=0, on_change=on_slider)
        if depth_or_saliency != "none":
            st.session_state["depth_mask"], st.session_state["saliency_mask"] =   predict_depth_and_saliency(effect_input)

            # apply black and white level transforms
            st.session_state["depth_mask"] = torch.clamp((st.session_state["depth_mask"] - mask_black_level) / (mask_white_level - mask_black_level), 0, 1)
            st.session_state["saliency_mask"] = torch.clamp((st.session_state["saliency_mask"] - mask_black_level) / (mask_white_level - mask_black_level), 0, 1)

        use_saliency_mask, use_depth_mask = False, False
        if depth_or_saliency == "depth":
            use_depth_mask = True
        elif depth_or_saliency == "saliency":
            use_saliency_mask = True
        invert_mask = st.checkbox("invert mask", value=False, on_change=on_slider)
        # choose from availalbe param masks or all
        active_param = st.selectbox("param", ["all"] + param_set, index=0, on_change=on_slider)
        # choose interpolation strength
        interpolation_strength = st.slider("interpolation strength", 0.0, 1.0, value=0.5, step=0.01, on_change=on_slider)
        interpolated_vp = vp.clone()

        if depth_or_saliency != "none":
            params = [active_param] if active_param != "all" else param_set
            for p in params:
                param_idx = effect.vpd.name2idx[p]
                current_param = vp[:, param_idx:param_idx + 1]
                preset = st.session_state["presets"][interpolate_with_preset] if "presets" in st.session_state else st.session_state["original_vp"]
                current_param = interpolate_between_params_with_masks(current_param, preset[:, param_idx:param_idx + 1],
                                                            use_saliency_mask, use_depth_mask,
                                                            invert_mask, invert_mask,
                                                            interpolation_strength, st.session_state)
                
                interpolated_vp[:, param_idx:param_idx + 1] = current_param
            commit_btn = st.button("commit")
            if commit_btn:
                vp = interpolated_vp
        

        if use_depth_mask:
            st.image(pil_resize_long_edge_to(np_to_pil(torch_to_np(st.session_state["depth_mask"])*255), 400), caption="depth mask")
        elif use_saliency_mask:
            st.image(pil_resize_long_edge_to(np_to_pil(torch_to_np(st.session_state["saliency_mask"])*255), 400), caption="saliency mask")
    with tab6:
        use_content_for_xdog = st.checkbox("use original content for xdog", value=st.session_state.get("use_content_for_xdog", False), on_change=on_slider)
        st.session_state["use_content_for_xdog"] = use_content_for_xdog
        configure_effect_xdog(effect, effect_input, use_content_for_xdog)


img_res = render_effect(effect, effect_input, interpolated_vp, firststage_config, pre_segmented_input)

st.session_state["result_vp"] = vp 
st.session_state["effect_input"] = effect_input
st.session_state["last_result"] = img_res

with coll1:
    # width = int(img_res.width * 500 / img_res.height)
    result_image_placeholder.image(img_res)  # , width=width)

if st.button("Clear All memoized functions"):
    # Clear values from *all* memoized functions:
    st.experimental_memo.clear()
    torch.cuda.empty_cache()

# a bit hacky way to return focus to top of page after clicking on images
components.html(
    f"""
        <p>{st.session_state.click_counter}</p>
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """,
    height=0
)
