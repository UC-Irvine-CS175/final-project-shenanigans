# Augmenting Image Datasets with GANs

## Introduction
While in space, astronauts are exposed to harmful cosmic radiation. By accurately categorizing radiation, scientists and engineers can design better shielding materials and develop appropriate countermeasures to mitigate the harmful effects of radiation on astronauts' health.

However, creating a comprehensive dataset for radiation classification is challenging due to the limited amount of available data. To overcome these limitations and improve the accuracy of the radiation classification model, the augmentation of the dataset with generated images can be a valuable approach.

## Getting Started

#### Setting Up the Environment
The environment can be setup by following the instructions in `setup/environments/README.md`.

#### Downloading the Dataset
The data comes from the [NASA BPS Microscopy Dataset](https://aws.amazon.com/marketplace/pp/prodview-6eq625wnwk4b6) which contains Fluorescence microscopy images of individual nuclei from mouse fibroblast cells, irradiated with Fe particles or X-rays. It is stored on an AWS S3 bucket and can be downloaded by running `src/data_utils.py`. The `dose_Gy_specifier` and `hr_post_exposure_val` variables can be changed to match the desired subset of the data.

## Models
This repository contains a vanilla GAN model at `src/gan.py` and a ResNet-101 model at `src/model/resnet101.py`.

## Credit
This was made by shenaniGANs ([Brandon Huynh](https://github.com/bdogetlauncher), [Olivia Ih](https://github.com/OliviaIh), [Dennis Lustre](https://github.com/dlustre), [Sharon Ma](https://github.com/sharonm6)).
