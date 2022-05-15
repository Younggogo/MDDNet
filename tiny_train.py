# -*- coding: utf-8 -*-
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from yolo3.tinymodel import preprocess_true_boxes, tiny_yolo_body, res2_yolo_body, yolo_loss, DY3TNet,yolo_body
from yolo3.utils import get_random_data
import tensorflow as tf
import os
import math
from functools import reduce
from operator import mul

def get_num_paras():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul,[dim.value for dim in shape], 1)
    return num_params

def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/'
    classes_path = 'model_data/voc_classes.txt' # 存储种类数量，即每一个种类的名称
    anchors_path = 'model_data/tiny_yolo_anchors.txt'  # 存储不同尺度特征图下，预先设置的anchor的尺寸
    class_names = get_classes(classes_path)  # 获取检测种类的名称
    anchors = get_anchors(anchors_path)  # 获取两个尺度下的anchor的尺寸 在原图上的尺寸

    input_shape = (224,416)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = create_model(input_shape, anchors, len(class_names) ) # 创建适用input_shape图像的模型
    train(model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir)

def train(model, annotation_path, input_shape, anchors, num_classes, log_dir='logs/'):
    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    tensorboard = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "best_weights.h5",
                                 monitor="val_loss",
                                 mode='min',
                                 save_weights_only=True,
                                 save_best_only=False, 
                                 verbose=1,
                                 period=1)

    callback_lists=[tensorboard,checkpoint]
    batch_size =10
    val_split = 0.05
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=10,
            initial_epoch=0, callbacks=callback_lists, workers=0, verbose=1)

    model.save_weights(log_dir + 'tiny-trained_weights.h5')
    total_parameters = get_num_paras()
    print(total_parameters)

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=False,
            weights_path='model_data/pretrained_weights.h5'):
    K.clear_session()
    image_input = Input(shape=(None, None, 25))                    #
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(math.ceil(h/{0:32, 1:16}[l]), math.ceil(w/{0:32, 1:16}[l]), \
        num_anchors//3, num_classes+5)) for l in range(2)]
    shape=(math.ceil(h/{0:32, 1:16}[0]), math.ceil(w/{0:32, 1:16}[0]), \
        num_anchors//3, num_classes+5)
    model_body = res2_yolo_body(image_input, num_anchors//3, num_classes) # MDDNet
    # model_body = tiny_yolo_body(image_input, num_anchors//3, num_classes)  # YOLOv3-tiny
    # model_body = DY3TNet(image_input, num_anchors // 3, num_classes) # DY3TNet
    # model_body = yolo_body(image_input, num_anchors // 3, num_classes) # YOLOv3
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',                                     #    错误  
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes) # 预处理锚框
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()