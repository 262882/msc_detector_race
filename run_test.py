#!/usr/bin/env python3
"""Start object detector test"""

import time
import json
import sys
import os
import cv2
import numpy as np
import onnxruntime as rt
sys.path.append(os.path.join(sys.path[0], 'tooling/'))
from helper_func import load_video

output_dir = 'results/'
model_dir = 'models/'
models_320 = ['nanodet.onnx']
models_416 = []
models_512 = []

test_sample = 'sample/detector_test.avi'

print("Welcome to the Object Detector Race")
run_time = time.perf_counter()

results = {}

#### ONNX settings ####
providers = ['CPUExecutionProvider']
sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
sess_options.intra_op_num_threads = 1
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

print("Prepare 320x320 environment")
data_320 = load_video(test_sample, 320)

for detector in models_320:

    print("Initialising detector")
    session = rt.InferenceSession(model_dir+detector, sess_options=sess_options, providers=providers)
    outname = [i.name for i in session.get_outputs()] 
    inname = [i.name for i in session.get_inputs()]
    print("Detector ready")

    time_log = []
    start_time = time.perf_counter()
    for frame in data_320:
        time_log.append(time.perf_counter()-start_time)

    print(time_log)

    results[detector[:-5] + '_320x320'] = time_log

with open("./" + output_dir + "race_results.json", 'w') as out_file:
            json.dump(results, out_file, indent=4)


'''
print("Initialising detector")

session = rt.InferenceSession('../models/nanodet.onnx', sess_options=sess_options, providers=providers)
outname = [i.name for i in session.get_outputs()] 
inname = [i.name for i in session.get_inputs()]
print("Detector ready")

print("Loading video stream")
cap = cv2.VideoCapture(sys.argv[1])
font = cv2.FONT_HERSHEY_DUPLEX
color = (0, 0, 0)
print("Loaded video stream")

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        blob = cv2.dnn.blobFromImage(_normalize(frame), #scalefactor=1/255,
                                        size = (320, 320), # Resolution multiple of 32
                                        swapRB=True, crop=False) 
        inp = {inname[0]:blob}
        layer_output = session.run(outname, inp)  # [(HxW)xC,(HxW)xstride] - Heatmaps                         

        # Record performance
        rate_fps = 1/(time.perf_counter()-run_time)
        run_time = time.perf_counter()
        frame = cv2.putText(frame, str(round(rate_fps))+" FPS",
                            ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

        # Perform inference
        coco_ind = 0  # Person
        stride = 8  # Stride: 8,16,32
        cell_count = 320//stride

        max_inds = np.argwhere(layer_output[0][:,coco_ind]>0.2)
        
        for ind in max_inds:

            # Find cell representation
            x = (ind%(cell_count**2))%cell_count
            y = (ind%(cell_count**2))//cell_count
            
            # Find max prediction as result
            box = np.max(np.reshape(layer_output[3][ind],(4,8)),axis=1)

            # Find pixel representation
            x = int((x+0.5)/cell_count*frame.shape[1])
            y = int((y+0.5)/cell_count*frame.shape[0])
            l,t,r,b = (np.copy(box)*stride*frame.shape[0]/320).astype("int")  # What is the scale here?

            #frame = cv2.circle(frame, (x,y), 10,  (0, 255, 0), 3)            
            frame = cv2.rectangle(frame, (x-l, y-t), (x+r, y+b),(0, 255, 0), 3)
        
        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break
        
cap.release() # When everything done, release the video capture object
cv2.destroyAllWindows() # Closes all the frames
'''

print("Complete")
