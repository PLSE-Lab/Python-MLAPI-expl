#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from joblib import dump, load
from sklearn.svm import SVC
from random import shuffle


# In[2]:


MODEL_NAME = 'FacialExpressionCNN'
LR = 0.001
IMG_SIZE = 48
def define_label(label):
    encoded_label = np.zeros(7)
    encoded_label[label] = 1
    return encoded_label


def create_data(df):
    data = []
    for i in range(len(df['pixels'])):
        imgInRow = list(df.iloc[i,1])
        res = ''
        for ch in imgInRow: res+=ch
        imgpixels = res.split()
        img = np.array([int(pixel) for pixel in imgpixels]).reshape(IMG_SIZE,IMG_SIZE)
        data.append([img,define_label(df.iloc[i,0])])
    return data


# In[3]:


data = pd.read_csv(r'../input/fer2013/fer2013.csv')
train_df = data.loc[data['Usage']=='Training']
valid_df = data.loc[data['Usage']=='PublicTest']
test_df = data.loc[data['Usage']=='PrivateTest']


# In[ ]:


print ('Train Data Loading Start....')
if (os.path.exists('train_data.npy')): train_data = np.load('train_data.npy')
else:
    train_data = create_data(train_df)
    np.save('train_data.npy',train_data)
shuffle(train_data)
print ('Train Data Loading End and has length:',len(train_data))


print ('Valid Data Loading Start....')
if (os.path.exists('valid_data.npy')): valid_data = np.load('valid_data.npy')
else:
    valid_data = create_data(valid_df)
    np.save('valid_data.npy',valid_data)
print ('Valid Data Loading End and has length:',len(valid_data))


print ('Test Data Loading Start....')
if (os.path.exists('test_data.npy')): test_data = np.load('test_data.npy')
else:
    test_data = create_data(test_df)
    np.save('test_data.npy',test_data)
print ('Test Data Loading End and has length:',len(test_data))


# In[ ]:


print ('Start Splitting.....')
X_train = np.array([pairXY[0] for pairXY in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [pairXY[1] for pairXY in train_data]

X_valid = np.array([pairXY[0] for pairXY in valid_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_valid = [pairXY[1] for pairXY in valid_data]

X_test = np.array([pairXY[0] for pairXY in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [pairXY[1] for pairXY in test_data]
print ('Splitted Succefully :D')


# In[ ]:


tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input, 64, 5, activation='relu')
pool1 = max_pool_2d(conv1, 2)
norm1 = local_response_normalization(pool1,5)

conv2 = conv_2d(norm1, 96, 5, activation='relu')
pool2 = max_pool_2d(conv2, 2)
norm2 = local_response_normalization(pool2,3)

conv3 = conv_2d(norm2, 256, 5, activation='relu')
pool3 = max_pool_2d(conv3,2)

conv4 = conv_2d(pool3, 256, 5, activation='relu')
fully_layer = fully_connected(conv4, 2048, activation='relu')
cnn_layers = fully_connected(fully_layer, 7, activation='softmax')
cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)


if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')

else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=1000,
          validation_set=({'input': X_valid}, {'targets': y_valid}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

