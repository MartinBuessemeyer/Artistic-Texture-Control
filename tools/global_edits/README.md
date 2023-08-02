# Global Editing Demo

This app demonstrates the editing capabilities of the Artistic Texture Control Framework
It optimizes the parameters of classical image processing filters to match a given style image.
The app provides different editing techniques, clustered by their category:
- global parameter tuning
- segmentation (painttransformer or SLIC)
- (re-) optimization. Optimize with different losses such L1, Gatys Loss, STROTTS Loss, CLIP Loss. 
- Prediction with arbitrary parameter prediction network
- Interpolation of parameters using depth and saliency masks


Make sure to have the packages specified in `requirements.txt` installed, the exact streamlit and streamlit drawable versions are required.
To run the demo, execute in the current directory:

`python -m streamlit run global_editing_streamlit_app.py`