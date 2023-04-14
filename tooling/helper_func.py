#!/usr/bin/env python3
"""Helper functions"""

import cv2
import numpy as np

def _normalize(img): 
    img = img.astype(np.float32) / 255
    MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    img = img - MEAN / STD
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

