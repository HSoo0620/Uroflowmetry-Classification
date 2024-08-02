import cv2
import numpy as np
import os

GREEN = (0, 255, 0)
Margin = 10

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


def main():
    origin_img = cv2.imread('01597119.jpg', cv2.IMREAD_GRAYSCALE)

    img = origin_img.copy()

    canny_img = canny(img, 50, 150)
    all_contour, one_contour, contours = get_contours(canny_img.copy())
    x_min, x_max, y_min, y_max= find_min_max(contours)

    sliced_img = origin_img[y_min-Margin:y_max+Margin,x_min-Margin:x_max+Margin]
    # slicing_y_index = int((y_max+Margin)/2)+Margin*5
    # sliced_img = origin_img[y_min-Margin:slicing_y_index,x_min-Margin:x_max+Margin]
    print("size of sliced image: ", sliced_img.shape)

    sliced_canny_img = canny(sliced_img, 50, 150)

    cv2.imshow('origin_img', origin_img)
    cv2.imshow('canny_img', canny_img)
    cv2.imshow('all_contour', all_contour)
    cv2.imshow('one_contour', one_contour)
    cv2.imshow('sliced_img', sliced_img)
    # cv2.imshow('sliced_canny_img', sliced_canny_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()