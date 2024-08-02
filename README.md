# HnN-Registration

Implement of "Uroflowmetry Classification"

## Overall Process
<img src = "https://github.com/user-attachments/assets/b5ff1784-eefc-487c-b726-9fe34a99da49" width="100%" height="100%">

## Pre-processing
Preprocessing first performs Canny edge detection, then finds the largest contour to crop the RoI.
Then perform resizing and normalization. 

<img src = "https://github.com/user-attachments/assets/a13836bb-ff7d-4fba-af23-ee0b039ee71b" width="100%" height="100%">

## Prerequisites
- [python==3.8.8](https://www.python.org/)  <br/>
- [pytorch==1.8.1](https://pytorch.org/get-started/locally/)

## Installation
The required packages are located in ```requirements```.

    pip install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    pip install -r requirement.txt

## Dataset

## Training

## Inference

## TODO
- [ ] main inference code
- [ ] Explanation dataset and Categorization
