# FCT-Pytorch
Pytorch implementation for The Fully Convolutional Transformer(FCT) 

## ACDC Dataset

Automated Cardiac Diagnosis Challenge Dataset available [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

Visualizations, analysis were performed to get a better understanding of the dataset.

## FCT Model

The Fully Convolutional Transformer (`FCT`) proposed by this [paper](https://arxiv.org/abs/2206.00566) is used in this project. The model is trained on the `ACDC` dataset using the `PyTorch` code variant available [here](https://github.com/kingo233/FCT-Pytorch).

Trained the model for 120 epochs which took around 3 hours on a `NVIDIA Tesla T4` GPU. Then, the model was saved for further analysis and auditing. Model is available [here](drive).

## Bias Audit for BMI

The `FCT` model was evaluated on the test set for 4 BMI categories.

## note

### This repo can:

- reproduces the origin aurhor's work on tensorflow.You need reference the original repo's issue that they only use ACDC train set(split ACDC/traning set into 7:2:1 train:validation:test).You can get dice 92.9
- Get about 90 dice on official test set if your train on the whole train set(using ACDC/training and test on ACDC/testing).


## training

1. Get ACDC dataset.And remember to delete `.md` file in your ACDC dataset folder
2. use `python main.py` to start training
