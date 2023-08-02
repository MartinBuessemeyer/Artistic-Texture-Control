# Evaluation

This directory contains the files used for the evaluation of our approach, i.e. ablation and pipeline comparison study
as well as our benchmarks of our
paper.

General Concept:
We use 10 style images and 20 random content images from MS COCO as our images for our studies.
For each run/study, we take the exact same images.

No script has command line arguments. The options are hard-coded at the top of the script as they should not be changed.

Example execution: `python -m ablation_study.arbitrary_style_pipeline_benchmark`

List of files:

- `arbitrary_style_for_ablation_study.py` -> Copy of the arbitrary style pipeline where filters can be skipped.
- `arbitrary_style_ablation_study.py` -> Script that executes the ablation study.
- `arbitrary_style_pipeline_benchmark.py` -> Benchmark script. (Runtime, GPU memory consumption)
- `calculate_param_mask_psnr` -> Calculates PSNR, TV Loss values for a pipeline run / configuration.
- `ppn_study_prediction.py` -> Script that generates the results of the study images using PPNs.
- `random_file_selection.sh` -> Shell script that randomly selects 20 images for MS COCO.



