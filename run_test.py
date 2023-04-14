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
from helper_func import load_video, std_preprocess, no_preprocess

output_dir = 'results/'
model_dir = 'models/'
models_320 = [
            ['nanodet.onnx', std_preprocess],
            ['ssd_mobilenet_v1_10.onnx', no_preprocess],
            ]
models_416 = [
            ['ssd_mobilenet_v1_10.onnx', no_preprocess],
            ]
models_512 = [
            ['ssd_mobilenet_v1_10.onnx', no_preprocess],
            ]

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
res = 320
data_set = load_video(test_sample, res)

for detector in models_320:

    print("Initialising detector")
    session = rt.InferenceSession(model_dir+detector[0], sess_options=sess_options, providers=providers)
    outname = [i.name for i in session.get_outputs()] 
    inname = [i.name for i in session.get_inputs()]
    print("Detector ready: " + detector[0])

    time_log = []
    start_time = time.perf_counter()
    for frame in data_set:
        inp = {inname[0]:detector[1](frame)}
        layer_output = session.run(outname, inp)
        time_log.append(time.perf_counter()-start_time)

    results[detector[0][:-5] + '_' + str(res) + 'x' + str(res)] = time_log
    print('Average speed: ', len(time_log)/time_log[-1])

print("Prepare 416x416 environment")
res = 416
data_set = load_video(test_sample, res)

for detector in models_416:

    print("Initialising detector")
    session = rt.InferenceSession(model_dir+detector[0], sess_options=sess_options, providers=providers)
    outname = [i.name for i in session.get_outputs()] 
    inname = [i.name for i in session.get_inputs()]
    print("Detector ready: " + detector[0])

    time_log = []
    start_time = time.perf_counter()
    for frame in data_set:
        inp = {inname[0]:detector[1](frame)}
        layer_output = session.run(outname, inp)
        time_log.append(time.perf_counter()-start_time)

    results[detector[0][:-5] + '_' + str(res) + 'x' + str(res)] = time_log
    print('Average speed: ', len(time_log)/time_log[-1])

print("Prepare 512x512 environment")
res = 512
data_set = load_video(test_sample, res)

for detector in models_512:

    print("Initialising detector")
    session = rt.InferenceSession(model_dir+detector[0], sess_options=sess_options, providers=providers)
    outname = [i.name for i in session.get_outputs()] 
    inname = [i.name for i in session.get_inputs()]
    print("Detector ready: " + detector[0])

    time_log = []
    start_time = time.perf_counter()
    for frame in data_set:
        inp = {inname[0]:detector[1](frame)}
        layer_output = session.run(outname, inp)
        time_log.append(time.perf_counter()-start_time)

    results[detector[0][:-5] + '_' + str(res) + 'x' + str(res)] = time_log
    print('Average speed: ', len(time_log)/time_log[-1])

with open("./" + output_dir + "race_results.json", 'w') as out_file:
            json.dump(results, out_file, indent=4)

print("Complete")
