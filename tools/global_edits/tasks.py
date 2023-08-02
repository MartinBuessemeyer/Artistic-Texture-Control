import base64
import copy
import datetime
import os
import sys
from io import BytesIO
from typing import Optional
import numpy as np
from types import SimpleNamespace
import requests
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
import time
import streamlit as st

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from parameter_optimization.parametric_styletransfer import single_optimize, get_options, get_loss_function
from parameter_optimization.strotss_org import strotss, pil_resize_long_edge_to
from effects import ArbitraryStyleEffect
from effect_helpers import create_segmentation
from helpers import torch_to_np, np_to_torch
WORKER_URL=""

brushstroke_config = {
    "lr_start": 0.01,
    "lr_stop": 0.00005,
    "lr_decay": 0.98,
    "lr_decay_start": 50,
    "n_iterations": 200,
    "local_params": True,
    "sigma_large": 1.5,
    "smoothing_steps": [],
    'experiment_name': 'test',
    'use_brushstroke': False,
    'brushstroke_weight': 0.15,
    'serial_rendering': False,
    'use_tv_loss': False,
    'tv_weight': 0.1,
    'use_hue_loss': False,
    'hue_weight': 0.1,
    "first_stage_type": "segmentation"
}
    

def optimize(effect, preset, style, result_image_placeholder, **kw_args):
    style = st.session_state["Style_im"]
    with st.spinner(text="Optimizing parameters.."):
        print("optimizing for user", st.session_state["user"])
        if st.session_state['Content_id'] == 'uploaded' or st.session_state['Style_id'] == 'uploaded':
            base_dir = ""
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            base_dir = os.path.join(dir_path, "precomputed", 
                                    st.session_state["effect_type"],
                                    st.session_state['Content_id'], 
                                    st.session_state['Style_id'])

        if not kw_args.get("segmented_target", False):
            kw_args["content_non_segmented"] = st.session_state['Content_im']

        if kw_args.get('different_style_image', False):
            style = kw_args['different_style_image']
                    
        segmentation_output = torch_to_np(create_segmentation(st.session_state["effect_input"], kw_args["firststage_config"]))
        content_segmented_input = Image.fromarray((segmentation_output * 255.0).astype(np.uint8)) 
        kw_args = {k: v for k, v in kw_args.items() if k not in  ["segmented_target", "different_style_image","firststage_config"] }

        optimize_params(effect, preset, content_segmented_input, style, result_image_placeholder, base_dir=base_dir, **kw_args)

def optimize_next(result_image_placeholder):
    result_image_placeholder.text("<- Custom content/style needs to be style transferred")
    queue_length = 0 #if not HUGGING_FACE else get_queue_length()
    if queue_length > 0:
        st.sidebar.warning(f"WARNING: Already {queue_length} tasks in the queue. It will take approx {(queue_length+1) * 5} min for your image to be completed.")
    else:
        st.sidebar.warning("Note: Optimizing takes up to 5 minutes.")
    optimize_button = st.sidebar.button("Optimize Style Transfer")
    if optimize_button:
        st.session_state["optimize_next"] = True
        st.experimental_rerun()
    else:
        if not "result_vp" in st.session_state:
            st.stop()
        else:
            return st.session_state["effect_input"], st.session_state["result_vp"]


def retrieve_for_results_from_server():
    task_id = st.session_state['current_server_task_id']
    vp_res = requests.get(WORKER_URL+"/get_vp", params={"task_id": task_id})
    image_res = requests.get(WORKER_URL+"/get_image", params={"task_id": task_id})
    if vp_res.status_code != 200 or image_res.status_code != 200:
        st.warning("got status for " + WORKER_URL+"/get_vp" + str(vp_res.status_code))
        st.warning("got status for " + WORKER_URL+"/image_res" + str(image_res.status_code))
        st.session_state['current_server_task_id'] = None
        vp_res.raise_for_status()
        image_res.raise_for_status()
    else:
        st.session_state['current_server_task_id'] = None
        vp = np.load(BytesIO(vp_res.content))["vp"] 
        print("received vp from server")
        print("got numpy array", vp.shape)
        vp = torch.from_numpy(vp).cuda()
        image = Image.open(BytesIO(image_res.content))
        print("received image from server")
        image = np_to_torch(np.asarray(image)).cuda()

        st.session_state["effect_input"] = image
        st.session_state["result_vp"] = vp


def monitor_task(progress_placeholder):
    task_id = st.session_state['current_server_task_id']

    started_time = time.time()
    retries = 3
    with progress_placeholder.container():
        st.warning("Do not interact with the app until results are shown - otherwise results might be lost.")
        progress_bar = st.empty()
        while True:
            status = requests.get(WORKER_URL+"/get_status", params={"task_id": task_id})
            if status.status_code != 200:
                print("get_status got status_code", status.status_code)
                st.warning(status.content)
                retries -= 1
                if retries == 0:
                    return
                else:
                    time.sleep(2)
                    continue
            status = status.json()
            print(status)
            if status["status"] != "running" and status["status"] != "queued" :
                if status["msg"] != "":
                    print("got error for task", task_id, ":", status["msg"])
                    progress_placeholder.error(status["msg"])
                    st.session_state['current_server_task_id'] = None
                    st.stop()
                if status["status"] == "finished":
                    retrieve_for_results_from_server()
                return
            elif status["status"] == "queued":
                started_time = time.time()
                queue_length = requests.get(WORKER_URL+"/queue_length").json()
                progress_bar.write(f"There are {queue_length['length']} tasks in the queue")
            elif status["progress"] == 0.0:
                progressed = min(0.5 * (time.time() - started_time) / 80.0, 0.5) #estimate 80s for strotts
                progress_bar.progress(progressed)
            else:
                progress_bar.progress(min(0.5 + status["progress"] / 2.0, 1.0))

            time.sleep(2)

def get_queue_length():
    queue_length = requests.get(WORKER_URL+"/queue_length").json()
    return queue_length['length']


def optimize_on_server(content, style, result_image_placeholder):
    content_path=f"/tmp/content-wise-uploaded{str(datetime.datetime.timestamp(datetime.datetime.now()))}.jpg"
    style_path=f"/tmp/content-wise-uploaded{str(datetime.datetime.timestamp(datetime.datetime.now()))}.jpg"
    asp_c, asp_s =  content.height / content.width, style.height / style.width
    if any(a < 0.5 or a > 2.0 for a in (asp_c, asp_s)):
        result_image_placeholder.error('aspect ratio must be <= 2')
        st.stop()

    content = pil_resize_long_edge_to(content, 1024)
    content.save(content_path)
    style = pil_resize_long_edge_to(style, 1024)
    style.save(style_path)
    files = {'style-image': open(style_path, "rb"), "content-image": open(content_path, "rb")}
    print("start-optimizing. Time: ", datetime.datetime.now())
    url = WORKER_URL + "/upload"
    task_id_res = requests.post(url, files=files)
    if task_id_res.status_code != 200:
        result_image_placeholder.error(task_id_res.content)
        st.stop()
    else:
        task_id = task_id_res.json()['task_id']
        st.session_state['current_server_task_id'] = task_id

    monitor_task(result_image_placeholder)



def optimize_params(effect, preset: list, content_segmented: PIL.Image, style: PIL.Image, 
    result_image_placeholder: st.container, loss: str = 'l1', start_from_preset: bool = True, 
    base_dir: str = "",  content_non_segmented: PIL.Image = None, nst_weight_content: float = 16.0, use_brushstrokeloss: bool = False,
    clipstyler_text_prompt: str = None, clipstyler_use_content_loss: bool = True, clipstyler_crop_size: int = 128,
    stylized_image_editing: bool = False, optim_steps: Optional[int] = None,):
    """
    Optimize the parameters of an effect to match the style of the style image and the content of the content image.

    :param effect: The effect whose parameters we want to optimize.
    :param preset: The initial parameters of the effect (disregarded if start_from_preset is False).
    :param content_segmented: The content image with segmentation applied.
    :param style: The style image. Not used for clipstyler loss, or l1 with an existing target.
    :param result_image_placeholder: The streamlit container to hold the loading bar
    :param loss: The loss function to use. Can be L1, STROTTS, GatysLoss or CLIPStyler 
    :param start_from_preset: Whether to start optimizing from the preset or from the current effect parameters
    :param base_dir: The base directory for loading the results.
    :param content_non_segmented: The content image without segmentation. Can be used as the content target for strotts loss instead of 'content'.
    :param clipstyler_text_prompt: The text prompt to use for clipstyler loss.
    :param clipstyler_use_content_loss: Whether to use the content loss for clipstyler loss.
    :param clipstyler_crop_size: The size of the patch crops.
    :param stylized_image_editing: Whether to use the content image as the style image instead of creating using strotts - useable with only l1 loss.

    :return: The optimized parameters.
    """
    result_image_placeholder.text("Executing NST to create reference image..")
    progress_bar = result_image_placeholder.progress(0.0)
 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(dir_path, "tmp_results", "reoptim", loss)
    os.makedirs(output_dir, exist_ok=True)

    config = copy.deepcopy(brushstroke_config)
    config["output_dir"] = output_dir

    if use_brushstrokeloss:
        config["use_brushstroke"] = True
        config["use_tv_loss"] = True

    if optim_steps is not None:
        config["n_iterations"] = optim_steps
    else:
        config["n_iterations"] = 300   if  not isinstance(effect, ArbitraryStyleEffect) else (300 if start_from_preset else 20)
    
    resize_to = 720
    content_save_path = os.path.join(output_dir, "content.jpg")
    content_segmented = pil_resize_long_edge_to(content_segmented, resize_to)
    if content_non_segmented is not None:
        content_non_segmented = pil_resize_long_edge_to(content_non_segmented, resize_to)
        content_non_segmented.save(content_save_path)
    else:
        content_segmented.save(content_save_path)
    config["img_size"] = resize_to
    config["content"] = content_save_path
    config["style"] = "" # path to style image, will be set later
    config["clipstyler_text"] = clipstyler_text_prompt
    config["loss"] = loss

    effect_input = np_to_torch(content_segmented).cuda()
    preset_or_vp = preset if start_from_preset else st.session_state["result_vp"]
    if not start_from_preset:
        preset_or_vp = F.interpolate(preset_or_vp, effect_input.shape[-2:])

    torch.cuda.empty_cache()

    parser = get_options()
    defaults = vars(parser.parse_args(["--content", content_save_path]))
    args = {key: parser.get_default(key) for key in defaults}
    # merge args and config dicts
    args.update(config)
    args = SimpleNamespace(**args)
    print("reoptimizing parameters with content target non-segmented = ", content_non_segmented is not None)

    style_path = None
    if loss != "CLIPStyler" and loss != "L1":
        style = pil_resize_long_edge_to(style, 1024)
        style_path = os.path.join(output_dir, "style.jpg")
        style.save(style_path)

    args.style = style_path

    if loss == 'L1':
        ref_save_path = os.path.join(base_dir, "reference.jpg")
        if stylized_image_editing:
            print("Stylized image editing - skipping style transfer")
            style.save(ref_save_path)
        if not os.path.exists(ref_save_path):
            reference = strotss(
                pil_resize_long_edge_to(
                    content_segmented if content_non_segmented is None else content_non_segmented, 1024
                ),
                pil_resize_long_edge_to(style, 1024), 
                content_weight=16.0, device=torch.device("cuda"), space="uniform")
            reference.save(ref_save_path) # save to precomputed dir
        else:
            reference = Image.open(ref_save_path)
        reference = pil_resize_long_edge_to(reference, resize_to)
        target = np_to_torch(reference).cuda()
    elif loss == 'STROTTS' or loss == "GatysLoss" or loss == 'CLIPStyler':
        target = torch.zeros_like(effect_input) #unused
    else:
        raise ValueError("Unknown loss function: " + loss)

    content = np_to_torch(content_segmented if content_non_segmented is None else content_non_segmented).cuda()
    loss_function = get_loss_function(args, content, effect_input).cuda()

    vp, content_img_cuda = single_optimize(
        effect.cuda(), preset_or_vp, loss_function, effect_input, target, args, 
        iter_callback=lambda i: progress_bar.progress(float(i) / config["n_iterations"])
    )

    st.session_state["effect_input"], st.session_state["result_vp"]  = content_img_cuda.detach(), vp.cuda().detach()
    return vp
