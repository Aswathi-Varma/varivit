## VariViT: A Vision Transformer for Variable Image Sizes

### Introduction

This repository contains code for the paper "VariViT: A Vision Transformer for Variable Image Sizes". The models are trained using two different approaches: one for ensuring consistent image sizes within the same batch, and another for handling varying image sizes.

### Setup

Before running the code, please follow these setup instructions:
1. Place the 3D extracted bounding boxes with three sizes in the `BASE_PATH` folder.
2. Update the `BASE_PATH` variable in the `brats_dataset/brats.py` and `egd_dataset/glioma.py` files accordingly.

### Configurations

The code uses argparse to handle command line arguments. However, configurations in the `config.ini` file can override any settings provided via the command line. Important settings in this file, particularly those in the `SETUP` section, determine parameters such as the number of epochs, output folder location, and logging directory.

### Training the Model

To train the model, follow these steps:
1. Run `K_fold_varivit_glioma_cbs.py` and `K_fold_varivit_brats_cbs.py` for ensuring consistent image sizes within the same batch.
2. Alternatively, use `K_fold_varivit_glioma_ga.py` and `K_fold_varivit_brats_ga.py` for handling varying image sizes. Adjust gradient accumulation iterations accordingly.
3. Baseline models, including vanilla ViT and ResNets, are also included for comparison purposes.

### Base Code Adaptation

The code base adapted from [Chinmay's GitHub repository](https://github.com/chinmay5/vit_ae_plus_plus).
