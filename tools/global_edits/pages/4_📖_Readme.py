import streamlit as st

st.title("Artistic Geometric Abstraction and Texture Control")

print(st.session_state["user"], " opened readme")
st.markdown("""
    ### How does it work?
    We provide a small stylization effect that contains several filters such as bump mapping or edge enhancement that can be optimized. The optimization yields so-called parameter masks, which contain per pixel parameter settings of each filter.
    
    ### Global Editing
    - On the first page select existing content/style combinations or upload images to optimize, which takes ~5min.
    - After the effect has been applied, use the parameter sliders to adjust a parameter value globally
    
    ### Local Editing
    - On the "apply preset" page, we defined several parameter presets that can be drawn on the image. Press "Apply" to make the changes permanent
    - On the " local editing" page, individual parameter masks can be edited regionally. Choose the parameter on the left sidebar, and use the parameter strength slider to either increase or decrease the strength of the drawn strokes
    - Strokes on the drawing canvas (left column) are updated in real-time on the result in the right column. 
    - Strokes stay on the canvas unless manually deleted by clicking the trash button. To remove them from the canvas after each stroke, tick the corresponding checkbox in the sidebar.

""")
