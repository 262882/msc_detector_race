#!/bin/bash

echo "Retrieving mobile ssd files"
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx &

echo "Retrieving yolox nano"
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx &

echo "Retrieving yolox tiny"
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx &

echo "Retrieving yolov3"
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx &

echo "Retrieving yolov3"
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx &

echo "Retrieving yolov4"
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx &

echo "Retrieving nanodet files"
wget https://github.com/hpc203/nanodet-opncv-dnn-cpp-python/raw/main/nanodet.onnx &

echo "Retrieving Faster R-CNN"
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx &

echo "Retrieving RetinaNet"
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx &
