# Controlling Geometric Abstraction and Texture for Artistic Images

### [Project Page](https://maxreimann.github.io/artistic-texture-editing/) | [Paper](https://arxiv.org/abs/2308.00148) | [Video](https://www.youtube.com/watch?v=Reip0iyY05U)

Official pytorch code for controlling texture and geometric abstraction in artistic images, e.g.,
results of neural style transfer.

<p align="center">
<img src="https://github.com/MartinBuessemeyer/Artistic-Texture-Control/assets/5698958/6f8ac2ef-48ab-4d00-87f4-0dbbfc358ae0" width="720">
</p>

[Controlling Geometric Abstraction and Texture for Artistic Images](https://maxreimann.github.io/artistic-texture-editing/)<br/>
[Martin Büßemeyer](https://www.researchgate.net/profile/Martin-Buessemeyer)\*<sup>1</sup>,
[Max Reimann](https://scholar.google.de/citations?user=Iz9UO7sAAAAJ&hl=en)\*<sup>1</sup>,
[Benito Buchheim](https://scholar.google.de/citations?hl=en&user=05SQiA8AAAAJ)<sup>1</sup>,
[Amir Semmo](http://asemmo.github.io/)<sup>2</sup>,
[Jürgen Döllner](https://hpi.de/forschung/fachgebiete/computergrafische-systeme.html)<sup>1</sup>,
[Matthias Trapp](https://scholar.google.de/citations?hl=en&user=zAHgRRQAAAAJ)<sup>1</sup> <br>
<sup>1</sup>Hasso Plattner Institute, University of Potsdam, Germany, <sup>2</sup>Digitalmasterpieces GmbH, Germany<br/>
*equal contribution <br/>
in [Cyberworlds 2023](https://cw2023.ieee.tn/)

## Main Idea


We propose a method to control the geometric abstraction (coarse features) and
texture (fine details) in artistic images (
such as results of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer)) separately from
another. We implement a stylization pipeline that geometrically abstracts images in its first stage and adds back the texture in
the second stage. The stages make use of the following abstraction and stylization methods:

- Geometric Abstraction:
    - Image Segmentation (we use [SLIC](DOI:10.1109/TPAMI.2012.120)) or
    - Neural Painters (we use the [PaintTransformer](https://doi.org/10.1109/ICCV48922.2021.00653))
- Texture Control:
    - Differentiable effect pipelines ([WISE](https://github.com/winfried-ripken/wise)). These represent an image style
      in the artistic control parameters of image filters. <br>
      We introduce a lightweight differentiable "Arbitrary Style Pipeline" which is capable of representing the
      previously "lost" texture, and is convenient to edit.

<img width="960" alt="tex_decomp" src="https://github.com/MartinBuessemeyer/Artistic-Texture-Control/assets/5698958/484a729d-fb52-4e38-a795-591381ca0721">

To acquire the filter parameters of the texture control stage, *texture decomposition* is executed by using either an
optimization-based approach that tunes parameters such that the effect output resembles a target image or by training a
network to do the same (both approaches were introduced by WISE).

After texture decomposition, various editing steps can be taken (see below).

## Features / Capabilities:

- Optimization-based texture decomposition:
    - default loss is L1 reconstruction loss to target image, but others can be used such as  [
      Gatys](https://doi.org/10.1109/CVPR.2016.265) style loss
    - slow, non-interactive, but agnostic to method
- Parameter Prediction Networks (PPNs):
    - The PPN predicts the parameters of the differentiable effect and is trained on one or multiple styles.
    - We demontraste this for Fast / Feed-Forward NST ([Johnson NST](https://arxiv.org/pdf/1603.08155.pdf)) and
      Arbitrary Style Transfer  [SANet](https://doi.org/10.1109/CVPR.2019.00603)
    - Benefits: fast, usable for interactive applications
    - Johnson NST is used by the provided editing prototype
- Editing and Flexibility:
    - Independent editing of geometric abstraction (edit first stage output) and texture (edit parameter masks of second
      stage)
    - Editing of Styles using text-prompts - here we use CLIP-based losses, similar
      to [CLIPStyler](https://github.com/cyclomon/CLIPstyler)
    - Mixing different styles: Use a different style for geometric abstraction and texture. Quality depends of style
      combination.
    - When using PPNs: Automatic adaption of texture / parameter masks after changing the geometric abstraction by
      repredicting the parameter masks
    - Some editing use-cases: (Re)move a misplaced style element, recover content information, locally / globally change
      the brushstroke texture.
    - The benefits can be explored by using the provided editing prototypes.

## Differentiable Effect Pipelines:

Pipelines included in this repository are:

- Arbitrary Style Pipeline (proposed stylization method, the default for this repository)
- XDoG (from WISE)

The modes of processing are:

- Parameter Optimization (Parametric Style Transfer and Checks for individual parameters)
- Parameter Predicition (Predict parameter masks using a PPN)

## NST Editing GUI

This GUI enables fine-granular editing of NSTs using our proposed arbitrary style pipeline. <br>

- It uses a parameter prediction network (PPN) trained on a single style to predict and edit parameters of a feedfoward
  style transfer.
- It lets a user edit the geometric abstraction and texture separately to influence specific aspects of the image

#### Setup

Install packages from requirements.txt

Run `python -m streamlit run tools/arbitrary_style_pipeline_editing_app.py`

#### NST Editing Workflow

We provide a near minimal set of editing tools to show the advantages of our method:

- Property Masks: To enable quick selection of important regions, we predict a depth and salience mask that can be
  thresholded.
- Color Palette Matching: Matches the color palette of the source region to the destination region via histogram
  matching. Adapt color schema from a different region.
- Color Interpolation: Interpolates the region with a color or the content image.
- Copy Region: Copies the source region to the center of the destination region. Copy a style element to another region.
- Change Level-of-Detail: Executes a re-segmentation on the selected region. Control geometric abstraction size.
- Re-Predict Parameter Masks: Repredicts the parameter masks for the texture stage. Thus the new masks adapt to the
  performed geometric abstraction edits.



https://github.com/MartinBuessemeyer/Artistic-Texture-Control/assets/5698958/021d7dcd-f644-4a34-b4d9-d3bc8981efd5


Please also see the [supplementary video](https://youtu.be/Reip0iyY05U?t=600) starting at 10:00 for a short demonstration of
an editing workflow to correct NST artifacts.

The geometric abstraction editing tools operate on segments predicted by SLIC. For artistic geometry editing, try out
the optimization-based prototype. <br>
Note: There is currently no 'revert' functionality in the prototype.
Further, the prototype needs a GPU with > 6 GB RAM for reasonably large images, as several CNNs are loaded into memory.

## Optimization-based Editing GUI

This app demonstrates the optimization-based synthesis and  editing capabilities including:
- global parameter tuning
- geometric abstraction control (painttransformer or SLIC)
- (re-) optimization. Optimize with different losses such as 
  - target image-based: L1
  - image-based style transfer: Gatys Loss, STROTTS Loss
  - text-based style transfer: CLIPStyler Loss. 
- Prediction with arbitrary parameter prediction network
- Interpolation of parameters using depth and saliency masks

Please also see the [supplementary video](https://youtu.be/Reip0iyY05U?t=428) starting at 7:05 for a short demonstration of the global editing app.

### Installing and running

Make sure to have the packages specified in `tools/global_edits/requirements.txt` installed, the exact streamlit and streamlit drawable versions are required.
To run the demo, execute in the `tools/global_edits/` directory:

`python -m streamlit run global_editing_streamlit_app.py`

## Optimization and training

### Parameter Optimization

Use the script: parameter_optimization/parametic_styletransfer.py
Example usage:

```
python -m parameter_optimization.parametric_styletransfer --content path/to/content_img --style path/to/style_img --img_size
512 --output_dir general/output/dir --experiment_name subfolder/in/output_dir
```

This will first execute a STROTTS NST with the given content and style. The result is used as input for our arbitrary
style pipeline. In the second step, the image segmentation will be used as the first pipeline stage and the parameter
masks for the texture stage are optimized.

Display script options: `python -m parameter_optimization.parametric_styletransfer --help`

Important Script Options:

- `--content path/to/content_img` Path to the content image that should be used.
- `--style path/to/style_img` Path to the style image that should be used.
- `--loss <one of: 'L1', 'CLIPStyler', 'GatysLoss', 'STROTTS'>` The loss that should be used as optimization goal.
- `--nst_variant <one of: 'STROTTS', 'JohnsonNST'>` Which NST variant should be used for the initial NST of the content
image.
- `--first_stage_type <one of: 'PaintTransformer', 'Segmentation'>` Which geometric abstraction should be used.
- `--clipstyler_text "My Text Prompt"` The text prompt for the CLIPStyler loss.
- `--n_iterations 500` Number of optimization steps to perform. We recommend at least 100.

### Pre-Trained Parameteter Prediction Network Weights

Currently available via
google-drive: https://drive.google.com/drive/folders/1mB6dhK-qzy6aESSKgMLBIrO9dTYc2eti?usp=sharing

### Training a Parameter Prediction Network

You can train your own Parameter Prediction Networks, if the pretrained models do not suit you. Keep in mind that this
requires at least two GPUs with ~20GB GPU RAM. We use
the [MS COCO](https://arxiv.org/abs/1405.0312) [dataset](https://cocodataset.org/#home) for content images during
training. Each Single Style Parameter Prediction Network needs a corresponding Johnson NST Network as this network
will generate the input for the arbitrary style pipeline.

Training a Johnson NST Network:

```
python -m parameter_prediction_network.johnson_nst.train--batch_size 16 --lr 5e-4 --logs_dir ./logs 
--style path/to/style --architecture johnson_instance_norm --dataset_path path/to/ms_coco --group_name <group_name> 
--img_size 256 --style_weight 5e10 --grad_clip 1e6 --epochs 12 --disable_logger
```

Training a Single Style Parameter Prediction Network:

```
python -m parameter_prediction_network.train --batch_size 16 --lr 5e-4 --logs_dir ./logs 
--style path/to/style_img --architecture johnson_instance_norm --dataset_path path/to/ms_coco --group_name <group_name> 
--img_size 256 --style_weight 5e10 --johnson_nst_model path/to/johnson_nst_weights --num_train_gpus <num_training_gpus - 1> --epochs 12
```

Important Script Options:

- `--content path/to/content_img` Path to the content image that should be used.
- `--style path/to/style_img` Path to the style image that should be used.
- `--img_size 256` Size of the content images during training.
- `--style_img_size 256` Size of the style image for loss calculation purposes. Changing the parameter might lead to bad
stylization.
- `--architecture johnson_instance_norm` Change the network architecture of the trained model. See list of available
models: `parameter_prediction_network/ppn_architectures/__init__`.
- `--num_train_gpus` Number of GPUs to use for training. If training PPN, must at least one lower as the actual GPU count
as one additional GPU will be used for the initial Johnson NST of the content images.
- `--johnson_nst_model` PPN only: Path to the Johnson NST weights, that should be used for the initial NST.

Training a Arbitrary Style Parameter Prediction Network works similar.
In addition to the [MS COCO](https://arxiv.org/abs/1405.0312) [dataset](https://cocodataset.org/#home),
you will need
the [Wikiart](https://doi.org/10.2308/iace-50038) [dataset](https://www.kaggle.com/datasets/steubk/wikiart).

## Code Acknowledgements

The code of this project is based on the following papers / repositories:

- [WISE](https://github.com/winfried-ripken/wise)
- [PaintTransformer](https://github.com/huage001/painttransformer)
- [STROTTS](https://github.com/nkolkin13/STROTSS)
- [Salience Estimation](https://github.com/weijun88/LDF)
- [Depth Estimation](https://github.com/isl-org/DPT)
- [SANet](https://github.com/glebsbrykin/sanet)

## Sources of Images in Experiments Folder

Here is the list of style images we used to train PPNs and in our experiments.
The `download_style_imgs.sh` script will download the files for you and put them into the designated
directory `experiments/target/popular_styles`.
The once used by us might be slightly different,
since we used a different source that depicts the same painting.

- [Rain Princess](https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/rain-princess-cropped.jpg)
- [The Great Wave](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/3213px-Tsunami_by_hokusai_19th_century.jpg)
- [Udnie](https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/udnie.jpg)
- [The Shipwreck of the Minotaur](https://img.huffingtonpost.com/asset/5bb235491f000039012379d6.jpeg)
- [Starry Night](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg)
- [Mosaic](https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/mosaic.jpg)
- [Candy](https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/candy.jpg)
- [Femme Nue Assise](https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg)
- [Delaunay](https://upload.wikimedia.org/wikipedia/commons/c/c9/Robert_Delaunay%2C_1906%2C_Portrait_de_Metzinger%2C_oil_on_canvas%2C_55_x_43_cm%2C_DSC08255.jpg)
- [The Scream](https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg)

# Questions?

Please do not hesitate to open an issue :).
