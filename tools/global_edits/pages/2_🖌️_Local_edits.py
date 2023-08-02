import torch.nn.functional as F
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas


PACKAGE_PARENT = '../../../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))




from effects.gauss2d_xy_separated import Gauss2DEffect
from effects import MinimalPipelineEffect, ArbitraryStyleEffect, OilPaintEffect
from effect_helpers import configure_effect_xdog
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings
# from demo_config import HUGGING_FACE

st.set_page_config(page_title="Editing Demo", layout="wide")

@st.cache(hash_funcs={MinimalPipelineEffect: id, ArbitraryStyleEffect: id, OilPaintEffect: id})
def local_edits_create_effect(effect_type):
    effect, preset, param_set = get_default_settings(effect_type)
    effect.enable_checkpoints()
    effect.cuda()
    return effect, param_set


effect, param_set = local_edits_create_effect(st.session_state["effect_type"])


@st.experimental_memo
def gen_param_strength_fig():
    cmap = matplotlib.cm.get_cmap('plasma')
    # cmap show
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(3, 0.1))
    fig.patch.set_alpha(0.0)
    ax.set_title("parameter strength", fontsize=6.5, loc="left")
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    return fig, cmap

cmap_fig, cmap = gen_param_strength_fig()

st.session_state["canvas_key"] = "canvas"
try:
    vp = st.session_state["result_vp"] if st.session_state["action"] != "switch_page_from_local_edits" else st.session_state["temp_vp"] 
    vp = vp.clone()
    org_cuda = st.session_state["effect_input"]
except KeyError as e:
    print("init run, certain keys not found. If this happens once its ok.")

configure_effect_xdog(effect, org_cuda, st.session_state["use_content_for_xdog"])

if st.session_state["action"] != "switch_page_from_local_edits":
    st.session_state.local_edit_action = "init"
    st.session_state["temp_vp"] = vp

st.session_state["action"] = "switch_page_from_local_edits" # on switchback, remember effect input

if "mask_edit_counter" not in st.session_state:
    st.session_state["mask_edit_counter"] = 1
if "initial_drawing" not in st.session_state:
    st.session_state["initial_drawing"] = {"random": st.session_state["mask_edit_counter"], "background": "#eee"}

def on_slider_change():
    if st.session_state.local_edit_action == "init":
        st.stop()
    st.session_state.local_edit_action = "slider"

def on_param_change():
    st.session_state.local_edit_action = "param_change"

active_param = st.sidebar.selectbox("active parameter: ", param_set + ["smooth"], index=2, on_change=on_param_change)

st.sidebar.text("Drawing options")
if active_param != "smooth":
    plus_or_minus = st.sidebar.slider("Increase or decrease param map: ", -1.0, 1.0, 0.0, 0.05,
                                      on_change=on_slider_change)
else:
    sigma = st.sidebar.slider("Sigma: ", 0.1, 10.0, 0.5, 0.1, on_change=on_slider_change)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 20, on_change=on_slider_change)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"), on_change=on_slider_change,
)

st.sidebar.text("Viewing options")
if active_param != "smooth":
    overlay = st.sidebar.slider("show parameter overlay: ", 0.0, 1.0, 0.8, 0.02, on_change=on_slider_change)
    st.sidebar.pyplot(cmap_fig, bbox_inches='tight', pad_inches=0)

st.sidebar.text("Update:")
realtime_update = st.sidebar.checkbox("Update in realtime", True)
clear_after_draw = st.sidebar.checkbox("Clear Canvas after each Stroke", False)
invert_selection = st.sidebar.checkbox("Invert Selection", False)


@st.experimental_memo
def greyscale_org(_org_cuda, content_id): #content_id is used for hashing
    # if HUGGING_FACE:
    #     wsize = 450
    #     img_org_height, img_org_width = _org_cuda.shape[-2:]
    #     wpercent = (wsize / float(img_org_width))
    #     hsize = int((float(img_org_height) * float(wpercent)))
    # else:
    longest_edge = 670
    img_org_height, img_org_width = _org_cuda.shape[-2:]
    max_width_height = max(img_org_width, img_org_height)
    hsize = int((float(longest_edge) * float(float(img_org_height) / max_width_height)))
    wsize = int((float(longest_edge) * float(float(img_org_width) / max_width_height)))

    org_img = F.interpolate(_org_cuda, (hsize, wsize), mode="bilinear")
    org_img = torch.mean(org_img, dim=1, keepdim=True) / 2.0
    org_img = torch_to_np(org_img)[..., np.newaxis].repeat(3, axis=2)
    return org_img, hsize, wsize

def generate_param_mask(vp):
    greyscale_img, hsize, wsize = greyscale_org(org_cuda, st.session_state["Content_id"])
    if active_param != "smooth":
        scaled_vp = F.interpolate(vp, (hsize, wsize))[:, effect.vpd.name2idx[active_param]]
        param_cmapped = cmap((scaled_vp + 0.5).cpu().numpy())[...,:3][0]
        greyscale_img = greyscale_img * (1 - overlay) + param_cmapped * overlay
    return Image.fromarray((greyscale_img * 255).astype(np.uint8))

def compute_results(_vp):
    if "cached_canvas" in st.session_state and st.session_state["cached_canvas"].image_data is not None:
        canvas_result = st.session_state["cached_canvas"]
        abc = np_to_torch(canvas_result.image_data.astype(np.float32)).sum(dim=1, keepdim=True).cuda()

        if invert_selection:
            abc = abc * (- 1.0) + 1.0

        img_org_width = org_cuda.shape[-1]
        img_org_height = org_cuda.shape[-2]
        res_data = F.interpolate(abc, (img_org_height, img_org_width)).squeeze(1)

        if active_param != "smooth":
            _vp[:, effect.vpd.name2idx[active_param]] += plus_or_minus * res_data
            _vp.clamp_(-0.5, 0.5)
        else:
            gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=5)
            gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=5)

            vp_smoothed = gauss2dx(_vp, torch.tensor(sigma).cuda())
            vp_smoothed = gauss2dy(vp_smoothed, torch.tensor(sigma).cuda())

            print(res_data.shape)
            print(_vp.shape)
            print(vp_smoothed.shape)
            _vp = torch.lerp(_vp, vp_smoothed, res_data.unsqueeze(1))


    if isinstance(effect, ArbitraryStyleEffect):
        with torch.no_grad():
            result_cuda = effect(st.session_state["first_stage_output"], _vp)
    else:
        with torch.no_grad():
            result_cuda = effect(org_cuda, _vp)

    _, hsize, wsize = greyscale_org(org_cuda, st.session_state["Content_id"])
    result_cuda = F.interpolate(result_cuda, (hsize, wsize), mode="bilinear")

    return Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8)), _vp

coll1, coll2 = st.columns(2)
coll1.header("Draw Mask:")
coll2.header("Live Result")

# there is no way of removing the canvas history/state without rerunning the whole program.
# therefore, giving the canvas a initial_drawing that differs from the canvas state will clear the background
def mark_canvas_for_redraw():
    print("mark for redraw")
    st.session_state["mask_edit_counter"] += 1 # change state of initial drawing
    initial_drawing = {"random": st.session_state["mask_edit_counter"], "background": "#eee"}
    st.session_state["initial_drawing"] = initial_drawing


with coll1:
    print("edit action", st.session_state.local_edit_action)
    if clear_after_draw and st.session_state.local_edit_action not in ("slider", "param_change", "init"):
        if st.session_state.local_edit_action == "redraw":
            st.session_state.local_edit_action = "draw"
            mark_canvas_for_redraw()
        else:
            st.session_state.local_edit_action = "redraw"

    mask = generate_param_mask(vp)
    st.session_state["last_mask"] = mask

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=stroke_width,
        background_image=mask,
        update_streamlit=realtime_update,
        width=mask.width,
        height=mask.height,
        initial_drawing=st.session_state["initial_drawing"],
        drawing_mode=drawing_mode,
        key=st.session_state.canvas_key,
    ) 

    if canvas_result.json_data is  None:
        print("stops")
        st.stop()

    st.session_state["cached_canvas"] = canvas_result

    print("compute result")
    img_res, vp = compute_results(vp)
    st.session_state["last_result"] = img_res
    st.session_state["temp_vp"] = vp
    if st.button("Commit"):
        st.session_state["result_vp"] = vp

    st.markdown("### Mask: " + active_param)

if st.session_state.local_edit_action in ("slider", "param_change", "init"):
    print("set redraw")
    st.session_state.local_edit_action = "redraw"

if "objects" in canvas_result.json_data and canvas_result.json_data["objects"] != []:
    print(st.session_state["user"], " edited local param canvas")       

print("plot masks")
texts = []
preview_masks = []
img = st.session_state["last_mask"]
for i, p in enumerate(param_set):
    idx = effect.vpd.name2idx[p]
    iii = F.interpolate(vp[:, idx:idx + 1] + 0.5, (int(img.height * 0.2), int(img.width * 0.2)))
    texts.append(p[:15])
    preview_masks.append(torch_to_np(iii))

coll2.image(img_res)  # , use_column_width="auto")
ppp = st.columns(len(param_set))
for i, (txt, im) in enumerate(zip(texts, preview_masks)):
    ppp[i].text(txt)
    ppp[i].image(im, clamp=True)

print("....")
