# FCT Bias Audit on ACDC

This project aims to perform a bias audit on the `FCT` model for different `BMI` and `Age` groups on the `ACDC` dataset.

## ACDC Dataset

Automated Cardiac Diagnosis Challenge Dataset available [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

Visualizations, analysis were performed to get a better understanding of the dataset.

## FCT Model

The Fully Convolutional Transformer (`FCT`) proposed by this [paper](https://arxiv.org/abs/2206.00566) is used in this project. The model is trained on the `ACDC` dataset using the `PyTorch` code variant available [here](https://github.com/kingo233/FCT-Pytorch).

Trained the model for 120 epochs which took around 3 hours on a `NVIDIA Tesla T4` GPU. Then, the model was saved for further analysis and auditing. Model is available [here](https://drive.google.com/file/d/12BPQm7GEcmTISYcYQXMJlQzwqohdDhb0/view?usp=share_link).

## Bias Audit for BMI

The `FCT` model was evaluated on the test set of 50 patients. Analysis will follow soon.

## Training

1. Get ACDC dataset and delete `.md` file in your ACDC dataset folder
2. use `python main.py` to start training
