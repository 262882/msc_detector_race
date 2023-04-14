#!/usr/bin/env python3
"""Helper functions"""

import cv2
import numpy as np

def std_normalize(img): 
    img = img.astype(np.float32) / 255
    MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    img = img - MEAN / STD
    return img

def nanodet_preprocess(img):
    '''
    return: NxCxHxW 
    '''
    blob = cv2.dnn.blobFromImage(std_normalize(img), 
                                size = img.shape[:-1],
                                swapRB=True, crop=False) 
    return blob

def yolox_preprocess(img):
    '''
    return: NxCxHxW 
    '''
    blob = cv2.dnn.blobFromImage(img,
                                size = img.shape[:-1], 
                                swapRB=True, crop=False) 
    return blob

def ssd_mobilenet_preprocess(img):
    '''
    return: NxHxWxC 
    '''
    blob = img[np.newaxis, ...]
    return blob

def yolov3_preprocess(img):
    '''
    return: NxCxHxW 
    '''
    image_data = np.array(img, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def yolov4_preprocess(img):
    '''
    return: NxHxWxC 
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img[np.newaxis, ...].astype(np.float32)
    return img

def load_video(path, out_res):

    batch_out = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            frame_re = cv2.resize(frame, (out_res, out_res), interpolation = cv2.INTER_AREA)
            batch_out.append(frame_re)

        else:
            break
            
    cap.release() # When everything done, release the video capture object
    return np.array(batch_out)

