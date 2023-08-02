from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html


def add_custom_js_and_css():
    # Define your javascript
    with open(Path(__file__).parent / 'costum_script.js', 'rt') as js_file:
        js_code = js_file.read()
    # Wrapt the javascript as html code
    my_html = f"<script defer>{js_code}</script>"

    # Execute your app
    # st.title("Javascript example")
    html(my_html)
