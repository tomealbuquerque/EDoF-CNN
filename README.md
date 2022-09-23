# EDoF-CNN
extended depth of field methods using CNNs for microscopy

##"Rethinking Low-Cost Microscopy Workflow:
Image Enhancement using Deep Based Extended Depth
of Field Methods"

by Tomé Albuquerque, Luís Rosado, Ricardo Cruz, Maria João M. Vasconcelos and Jaime S. Cardoso

## Introduction
Ordinal classific

## Usage

  1. Run datasets\prepare_{dataset name}.py to generate the data.
  2. Run train.py to train the models you want.
  3. Run evaluate.py to generate results table.


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
