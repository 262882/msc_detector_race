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

def nanodet_preprocess(inname, img):
    '''
    desired image shape: NxCxHxW 
    '''
    blob = cv2.dnn.blobFromImage(std_normalize(img), 
                                size = img.shape[:-1],
                                swapRB=True, crop=False) 
    
    return {inname[0]:blob}

def yolox_preprocess(inname, img):
    '''
    desired image shape: NxCxHxW 
    '''
    blob = cv2.dnn.blobFromImage(img,
                                size = img.shape[:-1], 
                                swapRB=True, crop=False) 
    return {inname[0]:blob}

def ssd_mobilenet_preprocess(inname, img):
    '''
    desired image shape: NxHxWxC 
    '''
    blob = img[np.newaxis, ...]
    return {inname[0]:blob}

def yolov3_preprocess(inname, img):
    '''
    desired image shape: NxCxHxW 
    '''
    image_data = np.array(img, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    image_size = np.array([image_data.shape[1], image_data.shape[0]], dtype=np.float32).reshape(1, 2)
    return {inname[0]:image_data, inname[1]:image_size}

def yolov4_preprocess(inname, img):
    '''
    desired image shape: NxHxWxC 
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    out_img = img[np.newaxis, ...].astype(np.float32)
    return {inname[0]:out_img}

def FRCNN_preprocess(inname, img):
    '''
    desired image shape: CxHxW 
    '''
    img = np.transpose(img, [2, 0, 1]) # HWC -> CHW
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(img.shape[0]):
        img[i, :, :] = img[i, :, :] - mean_vec[i]

    return {inname[0]:img.astype('float32')}

def retinanet_preprocess(inname, img):
    '''
    desired image shape: NxCxHxW 
    '''
    blob = cv2.dnn.blobFromImage(std_normalize(img), 
                                size = (640, 480),
                                swapRB=True, crop=True) 
    return {inname[0]:blob}

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

