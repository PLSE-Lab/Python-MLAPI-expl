#!/usr/bin/env python
# coding: utf-8

# # Base line model to show how to use tf with Object detection API for inference on the open images dataset
# 
# ## Overview
# This is my very first Kaggle kernel! So happy....
# This kernel is a proof of concept for using the tf object detection api on the data. I have noticed that many people are detered by Kaggle inability to access the outside world (like colab) so they are not using pre-trained models. The way to do that is to upload the pre trained models as private dataset. In this case I used the goodle trained model from their zoo. It is exptremely slow (44 seconds per image) so on the test set would take.... would take... mmmm very long time.
# 
# ## Data used in this kernel
# Labels for the open images dataset from https://github.com/tensorflow/models/blob/master/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt)
# 
# Models from:
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_14_10_2017.tar.gz
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz
# (I thought there were problems with the newer model but it turned out to be my mistake. The 2017 model is frcnn.pb, the 2018 is frcnn2.pb)
# 
# I have also uploaded the tf models git as a dataset but haven't played with it directly.
# 
# The model is faster rcnn with inception v2 as base.
# It is a very slow process but you can see how it performs on the leaderboard (with few tweaks) currently number 4

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
import cv2 as cv

#just double checking that GPU is actually reconized
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


#load labels
df = pd.read_csv('../input/openimages-labels/labels.csv')
df.set_index('id', drop=False)
print(df.head(5))
#not a pandas expert.... there must be a way to search for the id but this is good enough (id-1)
print(df.loc[4,'display'])


# In[ ]:


# Read the graph.
with tf.gfile.FastGFile('../input/frcnn-tf-model/frcnn2.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


# In[ ]:


#this takes a while...
with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


# In[ ]:


#do the actual inference
import time

with tf.Session() as sess:
    # Read and preprocess an image.
    img = cv.imread('../input/google-ai-open-images-object-detection-track/test/challenge2018_test/00febf2235f60610.jpg')
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    s = time.time()
    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    print("it took {} seconds for one image".format(time.time()-s))
    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    print(num_detections)
    preds_list=[]
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=3)
            className = df.loc[classId-1, 'display']
            print("classID {}, name {}, score {}".format(classId, className, score))
            preds_list.append(className)


# In[ ]:


from matplotlib import pyplot as plt
plt.imshow(img, aspect='auto')
plt.show()
preds_list

