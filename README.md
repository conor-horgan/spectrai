# spectrai
spectrai is an open-source deep learning framework designed to facilitate the training of neural networks on spectral data and enable comparison between different methods. Spectrai provides numerous built-in spectral data pre-processing and augmentation methods, neural networks for spectral data including spectral and spectral image denoising, spectral and spectral image classification, spectral image segmentation, and spectral image super-resolution.

## Contents
1. [Introduction](#introduction)
2. [Examples](#examples)
    - [Spectral Image Segmentation](#spectral-image-segmentation)
    - [Spectral Denoising](#spectral-denoising)
    - [Spectral Image SuperResolution](#spectral-image-superresolution)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Environment](#environment)
6. [Citation](#citation)

## Introduction
Spectral acquisition and imaging play important roles across numerous fields such as machine vision, remote sensing, and biomedical imaging. While there is much potential for spectral deep learning applications, existing deep learning frameworks and models for computer vision are largely oriented towards RGB images. Spectral data differs substantially from RGB images, however, and poses numerous requirements for which standard neural network architectures, data augmentations, and hyperparameter defaults are often unsuitable.  Application of existing deep learning models to spectral datasets thus requires careful modification and adaptation to make training on spectral data possible and effective. For example, while spectral augmentations (e.g., spectral flipping or shifting) may be applied, standard image augmentations (e.g., brightness or contrast changes) may introduce unwanted spectral distortions. Similarly, although 2D convolutional neural networks (CNNs) may be extended to multi-channel hyperspectral images, many spectral deep learning applications employ 1D or 3D CNNs, necessitating modification of existing 2D CNN architectures or the development of novel task-specific architectures. Lastly, the large size of spectral image hypercubes poses significant memory constraints which may require modification of network architectures and training hyperparameters (e.g., batch size, patch size, scaling, data augmentation) to enable effective single- or multi-GPU training.

We have thus developed spectrai, a deep learning framework designed specifically for spectral data. spectrai is built on the popular PyTorch library and includes baseline implementations of several networks for tasks including spectral calibration, spectral (image) classification, spectral denoising, spectral image segmentation, and spectral image super-resolution. In addition to a Python command line interface, spectrai provides a MATLAB graphical user interface (GUI) to guide non-expert users through deep learning training and inference procedures.

![Figure_1](/figures/figure_1.png)
spectrai MATLAB GUI.

## Examples
### Spectral Image Segmentation
We demonstrate spectral image segmentation using the recently developed AeroRIT hyperspectral image dataset ([dataset](https://github.com/aneesh3108/AeroRIT), [paper](https://arxiv.org/pdf/1912.08178.pdf)).

![Figure_2](/figures/figure_2.png)
AeroRIT Hyperspectral Image Semantic Segmentation.

### Spectral Denoising
We demonstrate spectral denoising using a dataset of Raman spectra of MDA-MB-231 human breast cancer cells ([dataset](), [paper](https://arxiv.org/abs/2009.13318)).

![Figure_3](/figures/figure_3.png)
Raman Spectral Denoising.

### Spectral Image SuperResolution
We demonstrate spectral image super-resolution using the HELICoiD dataset of intraoperative hyperspectral images of human brains ([dataset](https://hsibraindatabase.iuma.ulpgc.es/), [paper](https://doi.org/10.1109/ACCESS.2019.2904788)).

![Figure_4](/figures/figure_4.png)
HELICoiD Hyperspectral Image 8x Spatial Super-Resolution.

## Installation
### spectrai core
spectrai is hosted on the [Python Package Index (PyPI)](https://pypi.org/).
The latest version of spectrai can be installed using:
```bash
    >> pip install spectrai
```

*__Note:__ this will not include the spectrai.mlapp MATLAB application

### spectrai + MATLAB GUI
If you would like to use the spectrai MATLAB GUI, please follow the following steps:

Note that we provide the spectrai MATLAB GUI as an editable MATLAB application (.mlapp) to enable further open-source development. If desired, this can be exported to create an executable file. See [here](https://uk.mathworks.com/help/matlab/creating_guis/app-sharing.html) for details.
1. Install [MATLAB 2020b](https://uk.mathworks.com/downloads/web_downloads/download_release?release=R2020b)
2. Install [Python 3.8](https://www.python.org/downloads/) 

    *__Note:__ changes introduced in Python 3.9 are not compatible with MATLAB.
    
    *__Note:__ MATLAB will not work with Anaconda Python distributions.
    
3. In MATLAB, type the following to confirm the settings for MATLAB's Python interpreter (see [here](https://uk.mathworks.com/help/matlab/ref/pyenv.html) for details):
    ```bash
    >> pyenv
    ```
4. Download [spectrai](https://github.com/conor-horgan/spectrai). 
5. In MATLAB, navigate the working directory to:
    ```bash
    ~/spectrai/spectrai
    ```

## Usage
### spectrai core
spectrai can be run from the command line by typing:

1. To train a model from scratch:
    ```bash
    >> spectrai_train
    ```
2. To evaluate a pretrained model:
    ```bash
    >> spectrai_evaluate
    ```
3. To apply a pretrained model to a dataset:
    ```bash
    >> spectrai_apply
    ```

The above commands will operate on a default config file provided in spectrai/spectrai/configs. To apply spectrai to new data, spectrai settings can be assigned in one of two ways:
1. By default, settings are determined from .yml config files. Default config files are stored in [spectrai/spectrai/configs](https://github.com/conor-horgan/spectrai/tree/main/spectrai/configs). Users can develop custom config files, save them in the configs folder, and then specify them using the --config commandline flag, e.g.:
    ```bash
    >> spectrai_train --config custom_config.yml --verbose
    ```
2. Alternatively, users can start from a default config file and modify individual settings via the commandline as required, e.g.:
    ```bash
    >> spectrai_train --activation PReLU --batch_size 16 --data_normalization "Max Value"
    ```

Suitable values for different parameters are listed in the default config files.

### spectrai + MATLAB GUI
spectrai can alternatively be run from the MATLAB GUI. This GUI provides an easy-to-use interface that guides users through task selection, data loading, and data augmentation, as well as providing suggested initial default hyperparameter values. Additionally, the GUI provides visual feedback on losses and neural network outputs during training.

To run spectrai from the MATLAB GUI:

1. Open MATLAB and navigate the working directory to:
    ```bash
    ~/spectrai/spectrai
    ```
2. Type:
    ```bash
    >> appdesigner
    ```
3. Open spectrai.mlapp
4. Click Run (green play button on top menu bar)
5. Starting at Training/Inference do the following:
    1. In Training/Inference, select a training/inference option
    2. In Deep Learning Task, select a deep learning option
        - Based on the option selected, unsuitable or irrelevant fields will be disabled (grayed out)
    3. In Data Manager:
        - Select a data format
        - Load input/target data as appropriate (by default assumes data is stored in a single directory as separate files unless the Data as Directory checkbox is unticked
        - Select shuffle data, if desired
        - Select split data and enter data splits, if desired
    4. In Training Hyperparameters, set training hyperparameters
    5. In Network Hyperparameters, set network hyperparameters
    6. In Data Augmentation, select any data augmentations desired
    7. In Preprocessing, set any data preprocessing required (example effects of preprocessing can be viewed by selecting Preview Spectra or Preview Images)
    8. Press Train, Evaluate, or Apply as appropriate

## Environment
spectrai was implemented and tested in MATLAB R2020b and Python 3.8.8 using PyTorch 1.4.0 on a desktop computer with a Core i7-8700 CPU at 3.2 GHz (Intel), 32 GB of RAM, and a Titan V GPU (NVIDIA), running Windows 10 (Microsoft).
spectrai has not yet been extensively tested in other environments (it's on our to do list).

## Citation
If you find this code helpful in your work, please cite the following [paper](https://arxiv.org/):

[Conor C. Horgan](https://www.kcl.ac.uk/people/conor-horgan) and [Mads S. Bergholt](http://www.bergholtlab.com/), "spectrai: a deep learning framework for spectral data", arXiv 2021, [arXiv](https://arxiv.org/)
