# Uroflowmetry-Classification

Implement of "Uroflowmetry Classification"

## Overall Process
<img src = "https://github.com/user-attachments/assets/0adedc9d-3374-4b4f-900c-403d404ea77a" width="100%" height="100%">

## Pre-processing
Preprocessing first performs Canny edge detection, then finds the largest contour to crop the RoI.

Then perform resizing and normalization. 

<img src = "https://github.com/user-attachments/assets/a13836bb-ff7d-4fba-af23-ee0b039ee71b" width="100%" height="100%">

- For training, reference ```Find_RoI.py``` and ```img_preprocessing.ipynb``` .
 
## Prerequisites
- [python==3.8.8](https://www.python.org/)  <br/>
- [pytorch==1.8.1](https://pytorch.org/get-started/locally/)

## Installation
The required packages are located in ```requirements```.

    pip install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    pip install -r requirement.txt

## Dataset
<img src = "https://github.com/user-attachments/assets/c2c210ff-3d4c-40e4-8b6f-078446646cd6" width="100%" height="100%">

- Class는 총 6개(AUR, BPH, Normal, OAB, Stricture, Underactive bladder)로 3가지(Danger, Warning, Normal)로 분류하여 사용자에게 피드백 주는 것이 목표입니다. 

- AUR은 급성 요폐, BPH는 양성 전립선 비대증, OAB는 과민성 방광, Stricture는 요도 협착증, Underactive bladder(UB)는 저활동성 방광입니다.

- 분류 기준은 배뇨량이 100이하이고, 잔뇨량이 300이하인 경우 혹은 잔뇨량이 400 이하인 경우 AUR로 Danger에 속합니다.

- Stricture는 최고 요속 5이하일 경우, BPH는 최고 요속 10 이하일 경우 Danger에 속합니다.

- BPH와 UB는 는 delta Q 값이 6.5 이하인 경우와 PVR-R이 40%이상인 경우 Danger에 속합니다. (delta Q는 최고 요속에서 평균 요속을 뺀 값 입니다.)

    - (BPH와 UB의 요속그래프는 매우 유사하기 때문에 Classification이 어렵습니다. 따라서, 후처리로 Danger와 Warning을 결정합니다.)

- Danger가 아닌 질환은 Warning에 속합니다.
  
## Inference
```python
python Inference.py --data_csv test.csv \
```
최종 추론에 사용되는 Pre-Processing 기준은 다음과 같습니다. 
- Voided volume <= 100 and 잔뇨 >= 300
- 잔뇨 >= 400
- Voiding efficiency < 50%
이와 같은 경우 모델 추론 결과와 상관없이 위험 환자 입니다.

최종 추론에 사용되는 Post-Processing 기준은 다음과 같습니다. 
- Delta Q와 PVR-R을 이용하여 각 환자에 따라 위험과 경고로 분류합니다.
    - Delta Q는 최고요속 - 평균요속
    - PVR-R은 잔뇨 / (잔뇨+배뇨량)
- BPH : 최고요속 10 이하 | Detla Q 6.5 이하 | PVR-R 40% 이상 => 위험
- OAB : 모든 경우 => 경고
- Stricture : 최고요속 5이하 => 위험 
- UB : Detla Q 6.5 이하 | PVR-R 40% 이상 => 위험 

## Training
- For training, reference ```train.ipynb```.

## Testing
- For testing, reference ```test.ipynb```.

## Result
- reference ```/confusion_matrix.ipynb```.
<img src = "https://github.com/user-attachments/assets/a9cde425-7dd3-452f-9d92-410e842bcb61" width="50%" height="50%">

## TODO
- [x] main inference code
- [x] Explanation dataset and Categorization
