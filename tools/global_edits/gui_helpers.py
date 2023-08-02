import base64
from io import BytesIO
import requests
from PIL import Image 
import os

import streamlit as st
from st_click_detector import click_detector
import streamlit.components.v1 as components
import helpers.session_state as session_state


def clickable_image_row(urls):
    html_code = '<div class="column" style="display: flex; flex-wrap: wrap; padding: 0 4px;">'
    for url in urls:
        html_code += f"<a href='#' id='{url['id']}' style='padding: 0px 5px'><img height='160px' style='margin-top: 8px;' src='{url['src']}'></a>"
    html_code += "</div>"
    return click_detector(html_code)

def last_image_clicked(type="content", action=None):
    kw = "last_image_clicked" + "_" + type
    if action:
        session_state.get(**{kw: action})
    elif kw not in session_state.get():
        return None
    else:
        return session_state.get()[kw]

@st.experimental_memo
def get_image_urls(effect_type):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dir_path, "precomputed")
    content_ids = [x for x in os.listdir(os.path.join(base_path, effect_type)) if os.path.isdir(os.path.join(base_path, effect_type, x))]

    content_urls = []
    style_urls = []
    for content_id in content_ids:
        style_ids = [x for x in os.listdir(
            os.path.join(base_path, effect_type, content_id)) 
            if os.path.isdir(os.path.join(base_path, effect_type, content_id, x))]
        
        for style_id in style_ids:
            content_path = os.path.join(base_path, effect_type, content_id, content_id+".jpeg")
            style_path = os.path.join(base_path, effect_type, content_id, style_id, "style.jpg")
            if os.path.exists(content_path) and content_id not in [x["id"] for x in content_urls]:
                img = Image.open(content_path).convert('RGB')
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                src_decoded = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                content_urls.append({"id": content_id, "src": src_decoded})
            if os.path.exists(style_path) and style_id not in [x["id"] for x in style_urls]:
                img = Image.open(style_path).convert('RGB')
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                src_decoded = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                style_urls.append({"id": style_id, "src": src_decoded})
    return content_urls, style_urls


@st.experimental_memo
def retrieve_from_id(clicked, urls):
    src = [x["src"] for x in urls if x["id"] == clicked][0]
    # check if src is url or base64 and decode accordingly
    if src.startswith("http"):
        img = Image.open(requests.get(src, stream=True).raw)
    else:
        img = Image.open(BytesIO(base64.b64decode(src.split(",")[1])))
    return img, src