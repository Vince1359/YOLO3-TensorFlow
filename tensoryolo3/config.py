# -*- coding:utf-8 -*-
# @author: Vince1359
# @file: config.py
# @time: 2019/1/15 11:56

num_parallel_calls = 4
input_shape = 416
max_boxes = 20
jitter = 0.3
hue = 0.1
sat = 1.0
cont = 0.8
bri = 0.1
norm_decay = 0.99
norm_epsilon = 1e-3
pre_train = True
num_anchors = 9
num_classes = 80
training = True
ignore_thresh = .5
learning_rate = 0.001
train_batch_size = 10
val_batch_size = 10
train_num = 118287
val_num = 5000
Epoch = 50
obj_threshold = 0.3
nms_threshold = 0.5
gpu_index = "0"
log_dir = '../logs'
data_dir = '../model_data'
model_dir = '../test_model/model.ckpt-192192'
pre_train_yolo3 = False
yolo3_weights_path = '../weights/yolov3.weights'
darknet53_weights_path = '../weights/darknet53.weights'
anchors_path = '../weights/yolo_anchors.txt'
classes_path = '../weights/coco_classes.txt'
train_data_file = '/data0/dataset/coco/train2017'
val_data_file = '/data0/dataset/coco/val2017'
train_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_train2017.json'
val_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_val2017.json'
