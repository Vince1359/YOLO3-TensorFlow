# -*- coding:utf-8 -*-
# @author: Vince1359
# @file: detect.py
# @time: 2019/1/22 19:17

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from tensoryolo3 import config
from tensoryolo3.predict import YOLOPredictor
from tensoryolo3.utils import letterbox_image, load_weights

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


def detect(image_path, model_path, yolo_weights=None):
    """
    加载模型，进行预测
    :param image_path: 图片路径
    :param model_path: 模型路径
    :param yolo_weights: yolo训练好的weights文件
    :return:
    """
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)
    predictor = YOLOPredictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    boxes, scores, classes = predictor.predict(input_image, input_image_shape)
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                input_image: image_data,
                input_image_shape: [image.size[1], image.size[0]]
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='../font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for ii in range(thickness):
                draw.rectangle(
                    [left + ii, top + ii, right - ii, bottom - ii],
                    outline=predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.show()
        # image.save('./result1.jpg')


if __name__ == '__main__':
    image_file = './dog.jpg'
    detect(image_file, config.model_dir, config.yolo3_weights_path)
