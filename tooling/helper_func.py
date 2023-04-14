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
    blob = cv2.dnn.blobFromImage(std_normalize(img), #scalefactor=1/255,
                                size = img.shape[:-1],  # Resolution multiple of 32
                                swapRB=True, crop=False) 
    return blob

def yolox_preprocess(img):
    blob = cv2.dnn.blobFromImage(img, #scalefactor=1/255,
                                size = img.shape[:-1],  # Resolution multiple of 32
                                swapRB=True, crop=False) 
    return blob

def ssd_mobilenet_preprocess(img):
    blob = img[np.newaxis, ...]
    return blob

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

