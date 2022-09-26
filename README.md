# EDoF-CNN

### Rethinking Low-Cost Microscopy Workflow: Image Enhancement using Deep Based Extended Depth of Field Methods

by Tomé Albuquerque, Luís Rosado, Ricardo Cruz, Maria João M. Vasconcelos and Jaime S. Cardoso

## Introduction
Microscopic techniques in low-to-middle income countries are constrained
by the lack of adequate equipment and trained operators. Since light microscopy delivers crucial methods for the diagnosis and screening of numerous diseases, several efforts were made by the scientific community to create low-cost devices such as 3D-printed portable microscopes. Nevertheless, these types of devices present some drawbacks that directly affect image quality: the capture of the samples is done via mobile phones; more affordable lenses are usually used, leading to lower physical properties and images with lower depth-of-field; misalignments in the microscopic set-up regarding optical, mechanical, and illumination components are frequent, causing image distortions like chromatic aberrations. This work explores several pre-processing methods to tackle the presented issues, and a new workflow for low-cost microscopy is proposed. Additionally, two new deep learning models based on Convolutional Neural Networks are also proposed (EDoF-CNN-fast and EDoF-CNN-pairwise) to generate Extended Depth-of-Field (EDoF or EDF) images, being compared against state-of-the-art approaches. The models were tested using two different datasets: Cervix93 and a private dataset of cytology microscopic images captured with the µSmartScope device. Experimental results demonstrate that the proposed workflow can achieve state-of-the-art performance when generating EdoF images from low-cost microscopes.

Tested preprocessing **workflows**:<br />
<img src="https://github.com/tomealbuquerque/EDoF-CNN/blob/main/figures/workflow.PNG" width="600">

Comparasion between rigid and elastic aligment across an entire stack (motion image):<br />
Rigid Aligment             |  Elastic Alignment
:-------------------------:|:-------------------------:
![](https://github.com/tomealbuquerque/EDoF-CNN/blob/main/figures/rigid.gif)  |  ![](https://github.com/tomealbuquerque/EDoF-CNN/blob/main/figures/elastic.gif)

**EDOF-CNN-fast** schematic representation:<br />
<img src="https://github.com/tomealbuquerque/EDoF-CNN/blob/main/figures/EDoF-CNN-fast.PNG" width="500">


**EDOF-CNN-pairwise** schematic representation:<br />
<img src="https://github.com/tomealbuquerque/EDoF-CNN/blob/main/figures/EDoF-CNN-pairwise.PNG" width="500">

## Usage
  
  1. Run the aligment method (Chromatic/Rigid/elastic) if your dataset is misaligned.
  2. Run datasets\prepare_{dataset name}.py to generate the data.
  3. Run train.py to train the models you want.
  4. Run evaluate.py to generate results table.


## Code organization
```
EDoF-CNN
│ 
├─ README.md
├─ alignment_code
│  ├─ elastic_alignment
│  │  ├─ ANNlib-5.0.dll
│  │  ├─ LICENSE
│  │  ├─ NOTICE
│  │  ├─ TransformParameters.0.txt
│  │  ├─ TransformParameters.1.txt
│  │  ├─ crop_after_edof.py
│  │  ├─ elastix.exe
│  │  ├─ methods.py
│  │  └─ pyelastic_stacks_final.py
│  └─ rigid_aligment
│     ├─ full_preprocess_alignment.py
│     └─ utils_align.py
├─ dataset.py
├─ dataset
│  ├─ preprocess_cervix93.py
│  └─ preprocess_fraunhofer.py
├─ evaluate.py
├─ example_images
├─ figures
├─ metrics.py
├─ models.py
├─ results
│  ├─ results_cervix93
│  └─ results_fraunhofer
│     ├─ results_fraunhofer_elastic_only
│     ├─ results_fraunhofer_rigid
│     ├─ results_fraunhofer_rigid_elastic
│     └─ results_no_rgb_only_elastic
├─ test.py
├─ train.ps1
├─ train.py
├─ train.sh
├─ utils.py
└─ utils_files
   ├─ __init__.py
   ├─ automate_EDoF_imagej.py
   ├─ automatic_brightness_and_contrast.py
   └─ pytorch_ssim
      └─ __init__.py
```
  * **data:** Just one dataset is publicly available: 
    * Cervix93 — https://github.com/parham-ap/cytology_dataset/tree/master/dataset (accessed on: 23 January 2022);
  * **train.py:** train the different models with the different datasets, z-planes, and CNN models.
  * **evaluate.py:** generate latex tables with results using the output probabilities.

## Citation
If you find this work useful for your research, please cite our paper:


If you have any questions about our work, please do not hesitate to contact [tome.albuquerque@gmail.com](tome.albuquerque@gmail.com)
