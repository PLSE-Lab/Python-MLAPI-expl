#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

#utility script to create tfrecords
import wheat_tfrecord_util as tfutil

import tensorflow as tf
from tensorflow import keras

import wheat_util as util

#utility script for models
import wheat_yolov3

from wheat_yolov3 import (
    YoloV3, YoloLoss,
    yolo_anchors, yolo_anchor_masks
)
from wheat_util import freeze_all

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)


# In[ ]:


PATH = '../input/global-wheat-detection'
TRAIN_EXT = 'jpg'
TRAIN_TFREC_DIR = '../input/wheat-tfrecords'

SIZE = 416
BATCH_SIZE = 16
PRETRAINED_WEIGHTS = '../input/yolov3-tf-pretrained/yolov3.tf'


# In[ ]:


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32
    
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) *         tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
IMAGE_FEATURE_MAP = {
    'image/height': tf.io.VarLenFeature(tf.int64),
    'image/width': tf.io.VarLenFeature(tf.int64),
    'image/filename': tf.io.VarLenFeature(tf.string),
    'image/source_id': tf.io.VarLenFeature(tf.string),
    'image/encoded': tf.io.VarLenFeature(tf.string),
    'image/format': tf.io.VarLenFeature(tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64)
}

TEST_IMAGE_FEATURE_MAP = {
    'image/filename': tf.io.VarLenFeature(tf.string),
    'image/encoded': tf.io.VarLenFeature(tf.string),
    'image/format': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, size, data_type):
    if data_type!='test':
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        x_train = tf.io.decode_jpeg((x['image/encoded'].values[0]), channels=3)
        x_train = tf.image.resize(x_train, (size, size))
        labels = tf.cast(tf.sparse.to_dense(x['image/object/class/label']), tf.float32)
        y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                            tf.sparse.to_dense(x['image/object/bbox/ymin']),
                            tf.sparse.to_dense(x['image/object/bbox/xmax']),
                            tf.sparse.to_dense(x['image/object/bbox/ymax']),
                            labels], axis=1)

        paddings = [[0, wheat_yolov3.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
        y_train = tf.pad(y_train, paddings)
        
        return x_train, y_train
    else:
        x = tf.io.parse_single_example(tfrecord, TEST_IMAGE_FEATURE_MAP)
        x_test = tf.io.decode_jpeg((x['image/encoded'].values[0]), channels=3)
        x_test = tf.image.resize(x_test, (size, size))
        img_id = tf.sparse.to_dense(x['image/filename'])
        return x_test, img_id


def load_tfrecord_dataset(filepaths, size=416, n_readers=5,
                         n_read_threads=5, data_type='train'):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
                lambda filepath: tf.data.TFRecordDataset(filepath),
                                cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(lambda x: parse_tfrecord(x, size, data_type))
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset


# In[ ]:


anchors = yolo_anchors
anchor_masks = yolo_anchor_masks


# ### Prepare records for training

# In[ ]:


#To create TFRecord data for training & validation

# train_set, unique_img_data = tfutil.prepare_for_records(train_data, data_type='train')
# train_unique_set, val_unique_set = train_test_split(unique_img_data, test_size=0.20, random_state=42)

# train_unique_set = train_unique_set.reset_index().drop('index', axis=1)
# val_unique_set = val_unique_set.reset_index().drop('index', axis=1)

# tfutil.multiprocess_write_data_to_tfrecords(train_set,
#     train_unique_set,
#     num_list = [i for i in range(0,len(train_unique_set))],
#     filename_prefix='train'
#     )

# tfutil.multiprocess_write_data_to_tfrecords(train_set,
#     val_unique_set,
#     num_list = [i for i in range(0,len(val_unique_set))],
#     filename_prefix='val'
#     )


# In[ ]:


train_filepaths = os.path.join(TRAIN_TFREC_DIR,'train*')
val_filepaths = os.path.join(TRAIN_TFREC_DIR,'val*')
train_dataset = load_tfrecord_dataset(train_filepaths, 416)
val_dataset = load_tfrecord_dataset(val_filepaths, 416)


# In[ ]:


train_dataset = train_dataset.shuffle(buffer_size=512)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.map(lambda x, y: (
                transform_images(x, SIZE),
                transform_targets(y, anchors, anchor_masks, SIZE)))
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.map(lambda x, y: (
            transform_images(x, SIZE),
            transform_targets(y, anchors, anchor_masks, SIZE)))


# In[ ]:


learning_rate = 1e-3
NUM_CLASSES = 1
weights_num_classes = 80 #pretrained weights are trained with 80 classes
EPOCHS = 10


# In[ ]:


model = YoloV3(SIZE, training=True, classes=NUM_CLASSES)

#pretrained weights are from https://pjreddie.com/media/files/yolov3.weights
#Converted & uploaded them to use in this kernel. https://www.kaggle.com/tyagit3/yolov3-tf-pretrained
model_pretrained = YoloV3(
                SIZE, training=True, classes=weights_num_classes or NUM_CLASSES)
model_pretrained.load_weights(PRETRAINED_WEIGHTS)
model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
freeze_all(model.get_layer('yolo_darknet'))


# In[ ]:


optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
loss = [YoloLoss(anchors[mask], classes=NUM_CLASSES)
            for mask in anchor_masks]


# In[ ]:


model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=False)

callbacks = [
    ReduceLROnPlateau(verbose=1),
    EarlyStopping(patience=3, verbose=1),
    ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                    verbose=1, save_weights_only=True)
]

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=val_dataset)


# ### Test

# In[ ]:


tfutil.PATH = '../input/global-wheat-detection'
tfutil.TEST_IMAGES_PATH = os.path.join(tfutil.PATH,'test')
tfutil.TEST_EXT = 'jpg'
tfutil.TFREC_DIR = '/kaggle/working'

test_set, unique_img_test_data = tfutil.prepare_for_records(data_type='test')
test_set[:2]


# In[ ]:


tfutil.multiprocess_write_data_to_tfrecords(test_set,
    unique_img_test_data,
    num_list = [i for i in range(0,len(test_set))],
    filename_prefix='test',
    data_type='test'
    )


# In[ ]:


test_filepaths = os.path.join(tfutil.TFREC_DIR,'test*')
test_dataset = load_tfrecord_dataset(test_filepaths, size=416, data_type='test')


# In[ ]:


test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(lambda x, y: (
            transform_images(x, SIZE),y))


# In[ ]:


latest = tf.train.latest_checkpoint('/kaggle/working/checkpoints')
latest


# In[ ]:


yolo = YoloV3(classes=1)
yolo.load_weights(latest).expect_partial()
class_names = ['Wheat']


# In[ ]:


wheat_yolov3.yolo_iou_threshold = 0.5
wheat_yolov3.yolo_score_threshold = 0.5


# ### Visualise prediction for one test image

# In[ ]:


import matplotlib.pyplot as plt
import cv2
from skimage import io

for imgs,img_ids in test_dataset.take(1):
    boxes, scores, classes, nums = yolo(imgs)
    for i, image in enumerate(imgs):
        img_boxes = boxes[i].numpy()
        img_scores = scores[i].numpy() 
        img_boxes = img_boxes[img_scores >= wheat_yolov3.yolo_score_threshold]
        img_boxes = np.array(img_boxes)*1024 #convert relative points back to fit image size
        img_boxes = img_boxes.astype(int)
        img_scores = img_scores[img_scores >= wheat_yolov3.yolo_score_threshold]
        image_id = img_ids[i].numpy()[0]
        img_url = tfutil.TEST_IMAGES_PATH+'/'+image_id.decode("utf-8")+'.jpg'
        sample = io.imread(img_url)
#         sample = image.numpy()
        
        break
        
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in img_boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)


# Reference: [https://github.com/zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
