#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
# #         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


train_data_df=pd.read_csv("../input/global-wheat-detection/train.csv")
train_data_df.head()


# In[ ]:


image_id=[f"{i}.jpg" for i in train_data_df.image_id]
xmins,ymins,xmaxs,ymaxs=[],[],[],[]
for bbox in train_data_df.bbox:
    real_bbox=eval(bbox)
    
    xmin, ymin ,w ,h=real_bbox
    
    
    
    a=int(xmin+w)
    b=int(ymin+h)
    xmaxs.append(a)
    ymaxs.append(b)

    
    c=int(xmin)
    d=int(ymin)
    xmins.append(c)
    ymins.append(d)


# In[ ]:


data=pd.DataFrame()
data["filename"]=image_id
data["width"]=train_data_df.width
data["width"]=train_data_df.height

data["class"]=["wheat"]*len(image_id)

data["xmin"]=xmins
data["ymin"]=ymins

data["xmax"]=xmaxs
data["ymax"]=ymaxs


# In[ ]:


data.head()


# In[ ]:


data.to_csv("train_labels.csv",index=False)


# In[ ]:


pd.read_csv("/kaggle/working/train_labels.csv")


# **Create tfrecord for training**

# In[ ]:


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# In[ ]:


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from collections import namedtuple, OrderedDict


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'wheat':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id':bytes_feature(filename),
        'image/encoded':bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text':bytes_list_feature(classes_text),
        'image/object/class/label':int64_list_feature(classes),
    }))
    return tf_example


def main(csv_input, output_path, image_dir):
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    csv_input="train_labels.csv"
    output_path="train_label.record"
    image_dir="/kaggle/input/global-wheat-detection/train"
    main(csv_input, output_path, image_dir)


# **Now you can train efficientDet d0-d7**
# 
# Here is the link of different efficientDet models pretrained weights on coco dataset
# 
# 
# 
# EfficientDet-D0 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz](http://)
# 
# EfficientDet-D1 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1.tar.gz](http://)
# 
# EfficientDet-D2 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2.tar.gz](http://)
# 
# EfficientDet-D3* 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d3_softnms.tar.gz](http://)
# 
# EfficientDet-D4 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d4.tar.gz](http://)
# 
# 
# EfficientDet-D5 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d5.tar.gz](http://)
# 
# EfficientDet-D6 	[https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d6.tar.gz](http://)
# 
# EfficientDet-D7*    [https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d7.tar.gz](http://)
# 
# 
# 
# To know more about efficientDet checkout the google automl repo on github 
# 
# [https://github.com/google/automl/tree/master/efficientdet](http://)

# In[ ]:


cd "/kaggle/input/kerasversionefficientdet"


# In[ ]:


os.mkdir("/kaggle/working/model")


# In[ ]:


get_ipython().system('pip install pycocotools')


# After creating tf record files you have to put your .record file in to efficientdet repo else it will throw an error no match pattern which i don't know why if someone knows why this error occurs do let me know in comment

# In[ ]:


# uncomment for training
# !python main.py --mode=train --training_file_pattern=train.record --model_name=efficientdet-d3 --model_dir=/kaggle/working/model --model_name=efficientdet-d3 --ckpt=/kaggle/input/effiecientdetd3-10k-epoch-checkpoints --train_batch_size=4 --num_epochs=3000 --num_examples_per_epoch=16


# currently kaggle does not support tensorboard and i having trouble with showing losses on terminal because of tpu estimator.I have made the dataset which i created and repo publically so you can download repo with tfrecord and train it on google colab which supports tensorboard or locally if you have better gpu

# for Easy inferencing checkout my kernel
# [https://www.kaggle.com/ravi02516/tensorflow-efficientdet-d3-with-default-parameters](http://)
