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
- Class는 총 6개(AUR, BPH, Normal, OAB, Stricture, Underactive bladder)로 3가지(Danger, Warning, Normal)로 분류하여 사용자에게 피드백 주는 것이 목표입니다. 

- AUR은 급성 요폐, BPH는 양성 전립선 비대증, OAB는 과민성 방광, Stricture는 요도 협착증, Underactive bladder(UB)는 저활동성 방광입니다.

- 분류 기준은 배뇨량이 100이하이고, 잔뇨량이 300이하인 경우 혹은 잔뇨량이 400 이하인 경우 AUR로 Danger에 속합니다.

- Stricture는 최고 요속 5이하일 경우, BPH는 최고 요속 10 이하일 경우 Danger에 속합니다.

- BPH와 UB는 는 delta Q 값이 6.5 이하인 경우와 PVR-R이 40%이상인 경우 Danger에 속합니다. (delta Q는 최고 요속에서 평균 요속을 뺀 값 입니다.)

    - (요속그래프만 볼 경우, BPH와 UB는 전문가의 경우도 구분할 수 없습니다.)

- Danger가 아닌 질환은 Warning에 속합니다. 


## Training
- For training, reference ```train.ipynb```.

## Inference

## TODO
- [ ] main inference code
- [x] Explanation dataset and Categorization
