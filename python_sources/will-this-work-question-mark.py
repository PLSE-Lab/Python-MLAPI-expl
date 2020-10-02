#!/usr/bin/env python
# coding: utf-8

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

# My imports
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.image import imread
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


num_train_images = len(os.listdir("../input/train_images"))
print('Training image count: ', num_train_images)
num_test_images = len(os.listdir("../input/test_images"))
print('Testing image count: ', num_test_images)


# In[ ]:


raw_train_data = pd.read_csv('../input/train.csv').dropna().reset_index(drop=True)
print('Num of training data: ', len(raw_train_data))
raw_train_data.head()


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()


# In[ ]:


def encode2img(shape, encoded_pixels):
    img = np.zeros(shape)
    regions = len(encoded_pixels) // 2
    for r in range(regions):
        idx = 2 * r
        start_pixel = int(encoded_pixels[idx])
        span = int(encoded_pixels[idx + 1])
        col_cnt = shape[0]
        for offset in range(span):
            current_pixel = start_pixel + offset
            current_x = current_pixel % col_cnt
            current_y = current_pixel // col_cnt
            img[current_x, current_y] = 1
    return img


# In[ ]:


sample_train_image_filename = '000a4bcdd.jpg'
print('filename: ', sample_train_image_filename)
sample_train_image = imread('../input/train_images/' + sample_train_image_filename)
print('image size: ', sample_train_image.shape)
print('image color scale: ', np.max(sample_train_image), '~', np.min(sample_train_image))
plt.figure(figsize=(20,4))
imshow(sample_train_image)
plt.show()
for idx in range(len(raw_train_data)):
    img_id_class = raw_train_data.loc[idx]['ImageId_ClassId'].strip().split('_')
    img_id = img_id_class[0]
    img_class = img_id_class[1]
    if img_id == sample_train_image_filename:
        sample_train_mask_encode = raw_train_data.loc[idx]['EncodedPixels'].strip().split(' ')
        sample_train_mask = encode2img(sample_train_image.shape[:2], sample_train_mask_encode)
        sample_train_mask_img = sample_train_mask
        plt.figure(figsize=(20,4))
        imshow(sample_train_mask_img, cmap='gray')
        plt.show()


# In[ ]:


def generate_train_label(img_id, raw_label):
    train_img = imread('../input/train_images/' + img_id)
    class_cnt = 4
    label_shape = (train_img.shape[0], train_img.shape[1], class_cnt)
    label_img = np.zeros(label_shape)
    for idx in range(len(raw_label)):
        img_id_class = raw_label.loc[idx]['ImageId_ClassId'].strip().split('_')
        img_id = img_id_class[0]
        img_class = int(img_id_class[1]) - 1
        if img_id == sample_train_image_filename:
            train_mask_encode = raw_label.loc[idx]['EncodedPixels'].strip().split(' ')
            train_mask = encode2img(sample_train_image.shape[:2], sample_train_mask_encode)
            label_img[:,:,img_class] = train_mask
    normalized_train_img = np.array(train_img) / 255
    return normalized_train_img, label_img


# In[ ]:


train_set = []
label_set = []
for train_image_filename in tqdm(os.listdir("../input/train_images")[:30]):
    train_img, label_img = generate_train_label(train_image_filename, raw_train_data)
    train_set.append(train_img)
    label_set.append(label_img)


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1), activation='relu')
])
model.compile(
    optimizer=tf.train.RMSPropOptimizer(0.01),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy]
)
train_data = np.array(train_set)
train_label = np.array(label_set)
print('train data shape: ', train_data.shape, ', train label shape: ', train_label.shape)
model.fit(train_data, train_label, epochs=10, batch_size=32)


# In[ ]:


test_set = []
for test_image_filename in tqdm(os.listdir("../input/test_images")[:100]):
    test_img = np.array(imread('../input/test_images/' + test_image_filename))
    test_set.append(test_img)


# In[ ]:


test_data = np.array(test_set)
print('test data shape: ', test_data.shape)
result = model.predict(test_data, batch_size=1, verbose=1)
print('prediction shape: ', result.shape)


# In[ ]:


def mask2encoded(mask):
    col_cnt = mask.shape[0]
    row_cnt = mask.shape[1]
    pixel_cnt = col_cnt * row_cnt
    encoded_pixels = []
    start_col = -1
    start_row = -1
    span = 0
    counting = False
    pairs = []
    for current_pixel in range(pixel_cnt):
        current_row = current_pixel % col_cnt
        current_col = current_pixel // col_cnt
        if mask[current_row, current_col] > 0.5:
            if not counting:
                counting = True
                span = 1
                start_col = current_col
                start_row = current_row
            else:
                span += 1
        else:
            if counting:
                pairs.append(current_pixel)
                pairs.append(span)
            counting = False
            span = 0
            start_col = -1
            start_row = -1
    return encoded_pixels


# In[ ]:


predict_cnt = result.shape[0]
class_cnt = result.shape[3]
for p in tqdm(range(predict_cnt)):
    for c in range(class_cnt):
        current_mask = result[p,:,:,c]
        encoded_mask = mask2encoded(current_mask)


# In[ ]:




