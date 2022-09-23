# EDoF-CNN
extended depth of field methods using CNNs for microscopy

### Rethinking Low-Cost Microscopy Workflow: Image Enhancement using Deep Based Extended Depth of Field Methods

by Tomé Albuquerque, Luís Rosado, Ricardo Cruz, Maria João M. Vasconcelos and Jaime S. Cardoso

## Introduction
Access to microscopic techniques in low-to-middle income countries is constrained by the lack of adequate equipment and trained operators. Since light microscopy delivers crucial methods for the diagnosis and screening of numerous diseases, several efforts were made by the scientific community to create low-cost devices such as 3D printed portable microscopes. Nevertheless, these types of devices present some drawbacks: the capture of the samples is done by a mobile phone and the low quality of the mechanical parts can influence the alignment of the slides. Also the captured image used for the diagnostic has a very limited depth-of-field. Thus, to tackle all the presented issues several pre-processing methods are tested and a new workflow is proposed. Two new deep learning models based on Convolutional Neural Networks are also proposed (EDoF-CNN-fast and EDoF-CNN-pairwise) to generate Extended Depth-of-Field (EDoF or EDF) images which are compared against state-of-the-art approaches. The models were tested using two different datasets: Cervix93 and a private dataset from Fraunhofer Portugal AICOS contaning cytology miscrocopic images captured on $\mu$SmartScope. Experimental results demonstrate that the proposed workflow can achieve state-of-the-art performance when generating EdoF images from low-cost microscopes. The computational speed of the proposed workflow is fast enough for real-time clinical usage.

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
