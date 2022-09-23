# EDoF-CNN
extended depth of field methods using cnn

"Rethinking Low-Cost Microscopy Workflow: A New End-to-End Strategy"



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
