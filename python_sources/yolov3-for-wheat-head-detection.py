#!/usr/bin/env python
# coding: utf-8

# # Predict bounding box and score for wheat heads

# ## YOLOv3 weights
# ### Trained offline and shared as dataset https://www.kaggle.com/senthilskm/yolov3-weights
# ### How to add weights to input data?
# ### 1. Top right under Data
# ### 2. Click + Add data
# ### 3. Look for https://www.kaggle.com/senthilskm/yolov3-weights
# ### 4 Now it is added in /kaggle/input/yolov3-weights

# ## Use YOLOv3 TF2 implementation from https://github.com/YunYang1994/TensorFlow2.0-Examples

# In[ ]:


import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import yolov3.core.utils as utils
from yolov3.core.yolov3 import YOLOv3, decode


# ### Paths

# In[ ]:


WEIGHTS = '/kaggle/input/yolov3-weights/yolov3'
TEST_IMAGES = '/kaggle/input/global-wheat-detection/test/'


# In[ ]:


input_size = 416

input_layer = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights(WEIGHTS)
model.summary()

images = os.listdir(TEST_IMAGES)
image_id = []
pred_string = []

for image_path in images:
    original_image = cv2.imread(os.path.join(TEST_IMAGES, image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.2, method='nms')
    image_id.append(image_path.split('.')[0])
    string = ''
    for box in bboxes:
        x_min, y_min = box[:2]
        width, height = (box[2] - box[0], box[3] - box[1])
        score = round(box[4], 4)
        string = string + '{sc} {x_min} {y_min} {width} {height} '.format(
            sc=score, x_min=int(x_min), y_min=int(y_min), width=int(width), height=int(height))
    pred_string.append(string)

out_df = pd.DataFrame({'image_id': image_id,
                       'PredictionString': pred_string})
out_df.to_csv("submission.csv", index=False)

