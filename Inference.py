import pandas as pd
import os
from pathlib import Path
import torch
from PIL import Image
import cv2
import argparse
from model.ResNet import ResNet50
from natsort import natsorted
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

Margin = 10

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False

def resizing_and_binary_img(origin_img):
    img = origin_img.copy()
    resize_img = cv2.resize(img, (448, 224))
    resize_img_invert = cv2.bitwise_not(resize_img)
    ret, binary_img = cv2.threshold(resize_img_invert, 50, 255, cv2.THRESH_BINARY)
    return binary_img

def canny(img, min, max):
    canny_img = cv2.Canny(img, min, max)
    return canny_img

def sobel(img):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0) 
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(dx, dy)
    mag = np.clip(mag, 0, 255).astype(np.uint8) 

    dst = np.zeros(img.shape[:2], np.uint8) 
    dst[mag > 120] = 255 
    return dst

def find_min_max(cnt):
    
    x_min, x_max, y_min, y_max = np.ndarray.min(cnt[...,0]), np.ndarray.max(cnt[...,0]), np.ndarray.min(cnt[...,1]), np.ndarray.max(cnt[...,1])
            
    return x_min, x_max, y_min, y_max
    
def get_contours(img):
    contours, _= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    all_contour = img.copy()
    
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    one_contour = cv2.drawContours(img.copy(), cnt, -1, (0, 255, 0), 2)
    
    for certain_contour in contours:
        
        rect = cv2.minAreaRect(certain_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        all_contour = cv2.drawContours(all_contour, [box], -1, (0, 255, 0), 2)
        
    return all_contour, one_contour, cnt

def crop_img_opencv(origin_img):
    img = origin_img.copy()
    canny_img = canny(img, 70, 150)
    all_contour, one_contour, contours = get_contours(canny_img.copy())
    x_min, x_max, y_min, y_max= find_min_max(contours)

    crop_img = origin_img[y_min-Margin:y_max+Margin,x_min-Margin:x_max+Margin]

    return crop_img

class ImageTransform() :
    def __init__(self) :
        self.data_transform = {
            'val' : transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        
    def __call__(self, img, phase) :
        return self.data_transform[phase](img)

def soft_and_argmax(outputs):
    preds = F.softmax(outputs, dim=1)
    argmax_preds = np.argmax(preds.cpu()).tolist()
    
    return argmax_preds

def assign_class(outputs):
    if outputs == 0 :
        pred_class = 'BPH'
    elif outputs == 1 :
        pred_class = 'Normal'
    elif outputs == 2 :
        pred_class = 'OAB'
    elif outputs == 3 :
        pred_class = 'Stricture'
    elif outputs == 4 :
        pred_class = 'UB'
    return pred_class

def read_and_filter_csv(df):
    # 위험군 조건을 만족하는 데이터 필터링 및 출력
    danger_conditions = (
        ((df['Voided_volume'] <= 100) & (df['잔뇨'] >= 300)) |
        (df['잔뇨'] >= 400) |
        ((df['Voided_volume'] / (df['Voided_volume'] + df['잔뇨'])) < 0.5)
    )
    
    abnormal_data = df[danger_conditions]
    for _, row in abnormal_data.iterrows():
        condition = ""
        if (row['Voided_volume'] <= 100) & (row['잔뇨'] >= 300):
            condition = "Condition 1: Voided volume <= 100 and 잔뇨 >= 300"
        elif row['잔뇨'] >= 400:
            condition = "Condition 2: 잔뇨 >= 400"
        elif (row['Voided_volume'] / (row['Voided_volume'] + row['잔뇨']) < 0.5):
            condition = "Condition 3: Voiding efficiency < 50%"
        # print(f"File: {row['파일명']}, Category: {row['분류']}, Condition: {condition}")

    # 정상 데이터만 필터링
    normal_data = df[~danger_conditions]
    return normal_data

def pre_filter_danger(df):
    ''' 
    [다음 조건일 경우 위험]
    1. Condition: Voided volume <= 100 and 잔뇨 >= 300" 
    2. Condition: 잔뇨 >= 400" 
    3. Condition: Voiding efficiency < 50%"
    아닐시 모델추론 후 판단
    '''
    
    voiding_efficiency = (df['Voided_volume'] / (df['Voided_volume'] + df['잔뇨']))
    if ((df['Voided_volume'] <= 100) & (df['잔뇨'] >= 300)) or (df['잔뇨'] >= 400) or voiding_efficiency < 0.5:
        result = 'red'
    else :
        result = 'Unknown'
    return result        
        
def post_processing(pred_class, df):
    '''
    Delta_Q : 최고요속 - 평균요속 (Maximum_flow - Average_flow)
    PVR-R : 잔뇨 / 잔뇨+Voided_volume
    '''
    
    '''
    [각 클래스에 따라 위험, 경고로 분류]
    - BPH : 최고요속 10 이하 | Detla Q 6.5 이하 | PVR-R 40% 이상 => 위험 아닐시 경고
    - OAB : 모든 경우 => 경고
    - Stricture : 최고요속 5이하 => 위험 아닐시 경고
    - UB : Detla Q 6.5 이하 | PVR-R 40% 이상 => 위험 아닐시 경고
    '''
    Delta_Q = df['Maximum_flow'] - df['Average_flow']
    PVR_R = df['잔뇨'] / (df['잔뇨'] + df['Voided_volume'])
    result = 'green'
   
    if pred_class == 'BPH' :
        if (df['Maximum_flow'] <= 10) or (Delta_Q <= 6.5) or (PVR_R >= 0.4) :
            result = 'red'
        else :
            result = 'yellow'
        
    elif pred_class == 'OAB' :
        result = 'yellow'
    
    elif pred_class == 'Stricture' :
        if (df['Maximum_flow'] <= 5):
            result = 'red'
        else :
            result = 'yellow'
        
    elif pred_class == 'UB':
        if (Delta_Q <= 6.5) or (PVR_R >= 0.4) :
            result = 'red'
        else :
            result = 'yellow'
    
    return result

def run_classification(df, args):
    with torch.no_grad() : 
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # preprocessing setting
        yolov5_weights = args.detection_model_pt
        RoI_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_weights).to(device)
        roi_imgsz=(640, 640)
        
        # classification setting
        pt_dir = args.classification_model_pt
        uroflow_model = ResNet50(num_classes=5, channels=1).to(device)
        best_model = torch.load(pt_dir + natsorted(os.listdir(pt_dir))[-1])
        print('Model: {} loaded!'.format(natsorted(os.listdir(pt_dir))[-1]))
        uroflow_model.load_state_dict(best_model)
        uroflow_model.eval()
        transform = ImageTransform()
        
        for idx, row in df.iterrows():
            # 1. 고위험 분류
            pre_filter_result = pre_filter_danger(row)
            if pre_filter_result == 'red' :
                pred_color = pre_filter_result
                print(idx, ' pred result: ', pred_color, ' GT: ',row['Category'])
            else :
                # (이미지 로드)
                img_path = row['image_path']
                img = Image.open(img_path).convert('RGB')
                origin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # 2. Yolov5를 통한 RoI Crop
                if args.roi_crop_using_yolo :
                    crop_result = RoI_detection_model(img, size=roi_imgsz, augment=False)
                    box = crop_result.pred[0][0][:4].tolist()
                    x1, y1, x2, y2 = map(int, box)
                    crop_img = origin_img[y1:y2, x1:x2]
                else : 
                    crop_img = crop_img_opencv(origin_img)
                # 3. 추가 전처리
                preprocessed_img = resizing_and_binary_img(crop_img)
                preprocessed_img = transform(preprocessed_img, phase='val').unsqueeze(0).to(device)
                # 4. Classification
                output = uroflow_model(preprocessed_img)
                output = soft_and_argmax(output)
                pred_class = assign_class(output)
                # 5. post processing
                pred_color = post_processing(pred_class, row)
                print(idx, ' pred result:', pred_class, pred_color, ' GT: ',row['분류'], row['Category'])

        return pred_color

def main(args):
    df = pd.read_csv(args.Data_csv)
    pred_color = run_classification(df, args)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_csv', type=str, default='test_data.csv',
                        help='csv directory')
    parser.add_argument('--detection_model_pt', type=str, default='./best.pt',
                        help='detection model pre train directory')
    parser.add_argument('--classification_model_pt', type=str, default='./experiment/pre_crop_204/',
                        help='classification model pre train directory')
    parser.add_argument('--roi_crop_using_yolo', type=str2bool, default='True',
                        help='choose crop methods True: yolov5 False: opencv')
    
    args = parser.parse_args()
    main(args)
