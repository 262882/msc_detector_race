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
from helper_func import load_video, nanodet_preprocess, ssd_mobilenet_preprocess, yolox_preprocess
from helper_func import yolov3_preprocess, yolov4_preprocess, FRCNN_preprocess, retinanet_preprocess

output_dir = 'results/'
model_dir = 'models/'
models_320 = [
            ['nanodet-m.onnx', nanodet_preprocess],
            ['nanodet-plus-m-1.5x_320.onnx', nanodet_preprocess],
            ['ssd_mobilenet_v1_12.onnx', ssd_mobilenet_preprocess],
            #['FasterRCNN-10.onnx', FRCNN_preprocess],
            ]
models_416 = [
            ['nanodet-plus-m-1.5x_416.onnx', nanodet_preprocess],
            ['ssd_mobilenet_v1_12.onnx', ssd_mobilenet_preprocess],
            ['yolox_nano.onnx', yolox_preprocess],
            ['yolox_tiny.onnx', yolox_preprocess],
            ['tiny-yolov3-11.onnx', yolov3_preprocess],
            ['yolov3-10.onnx', yolov3_preprocess],    
            ['yolov4.onnx', yolov4_preprocess],
            ['FasterRCNN-10.onnx', FRCNN_preprocess],
            ]
models_512 = [
            ['ssd_mobilenet_v1_12.onnx', ssd_mobilenet_preprocess],
            #['FasterRCNN-10.onnx', FRCNN_preprocess],
            ]

test_sample = 'sample/detector_test.avi'

print("Welcome to the Object Detector Race")
run_time = time.perf_counter()

results = {}
model_batches = [
                [models_320, 320],
                [models_416, 416],
                [models_512, 512],
]

#### ONNX settings ####
providers = ['CPUExecutionProvider']
sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
sess_options.intra_op_num_threads = 1
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

for model_batch in model_batches:

    res = model_batch[1]

    print("Prepare " + str(res) + "x"+ str(res), "environment")
    data_set = load_video(test_sample, res)

    for detector in model_batch[0]:

        print("Initialising detector")
        session = rt.InferenceSession(model_dir+detector[0], sess_options=sess_options, providers=providers)
        outname = [i.name for i in session.get_outputs()] 
        inname = [i.name for i in session.get_inputs()]
        print("Detector ready: " + detector[0])

        time_log = []
        start_time = time.perf_counter()

        for frame in data_set:
            inp = detector[1](inname, frame)
            layer_output = session.run(outname, inp)
            time_log.append(time.perf_counter()-start_time)

        model_size_mb = os.stat(model_dir+detector[0]).st_size/(1024 * 1024)
        results[detector[0][:-5] + '_' + str(res) + 'x' + str(res)] = {'time': time_log, 'model_size':model_size_mb}
        print('Average speed: ', len(time_log)/time_log[-1])
        print('Model size: ', model_size_mb)

with open("./" + output_dir + "race_results.json", 'w') as out_file:
            json.dump(results, out_file, indent=4)

print("Complete")
