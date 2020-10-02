#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize
# from skimage.transform import AffineTr
from random import randint, choice, shuffle
from math import ceil
import matplotlib.pyplot as plt
import sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
batch_size = 16
epoch = 50
data_path = '../input'
train_ids = next(os.walk(data_path + '/stage1_train/'))[1]
shuffle(train_ids)
test_ids = next(os.walk(data_path + '/stage1_test/'))[1]
X = tf.placeholder(tf.float32, shape=(batch_size, IMG_HEIGHT,
                IMG_WIDTH, IMG_CHANNELS))
y = tf.placeholder(tf.float32, shape=(batch_size, IMG_HEIGHT,
                IMG_WIDTH, 1))


# In[ ]:


def get_data(id_):
    path = data_path + '/stage1_train/' + id_ 
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        res = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                  preserve_range=True)
        mask_ = np.expand_dims(res, axis=-1)
        mask = np.maximum(mask, mask_)
    return img, mask


# In[ ]:


def convolution(x, y_dim, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02,
                activation=tf.nn.relu, name='conv2d'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], y_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, s_h, s_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [y_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if activation:
            return activation(conv)
        return conv
    
def deconvolution(x, output_shape, k_h=2, k_w=2, s_h=2, s_w=2, std_dev=0.015,
                  activation=tf.nn.relu, name='deconv2d'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weight', [k_h, k_w, output_shape[-1], x.get_shape()[-1]],
                            tf.float32, initializer=tf.truncated_normal_initializer(stddev=std_dev))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape, [1, s_h, s_w, 1])
        b = tf.get_variable('biases', [output_shape[-1]], 
                            initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        if activation:
            return activation(deconv)
        return deconv
        


# In[ ]:


def build():
    ## Sector 1
    conv_1 = convolution(X, 16, name='conv_1')
    drop_1 = tf.nn.dropout(conv_1, 0.06)
    conv_2 = convolution(drop_1, 16, name='conv_2')
    pool_1 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Sector 2
    conv_3 = convolution(pool_1, 32, name='conv_3')
    drop_2 = tf.nn.dropout(conv_3, 0.06)
    conv_4 = convolution(drop_2, 32, name='conv_4')
    pool_2 = tf.nn.max_pool(conv_4, ksize=[1,1,1,1], strides=[1,2,2,1], padding='SAME')
    
    # Sector 3
    conv_5 = convolution(pool_2, 64, name='conv_5')
    drop_3 = tf.nn.dropout(conv_5, 0.06)
    conv_6 = convolution(drop_3, 64, name='conv_6')
    pool_3 = tf.nn.max_pool(conv_6, ksize=[1,1,1,1], strides=[1,2,2,1], padding='SAME')
    
    # Sector 4
    conv_7 = convolution(pool_3, 128, name='conv_7')
    drop_4 = tf.nn.dropout(conv_7, 0.06)
    conv_8 = convolution(drop_4, 128, name='conv_8')
    pool_4 = tf.nn.max_pool(conv_8, ksize=[1,1,1,1], strides=[1,2,2,1], padding='SAME')
    
    # Base
    conv_9 = convolution(pool_4, 256, name='conv_9')
    conv_10 = convolution(conv_9, 256, name='conv_10')
    
    # Sector 6
    conv_11 = tf.concat([conv_8, deconvolution(conv_10,
                                               [batch_size, 32, 32, 128], name='deconv_1')], 3)
    conv_12 = convolution(conv_11, 128, name='conv_12')
    conv_13 = convolution(conv_12, 128, name='conv_13')
    
    # Sector 7
    conv_14 = tf.concat([conv_6, deconvolution(conv_13, 
                                               [batch_size, 64, 64, 64], name='deconv_2')], 3)
    conv_15 = convolution(conv_14, 64, name='conv_15')
    conv_16 = convolution(conv_15, 64, name='conv_16')
    
    # Sector 8
    conv_17 = tf.concat([conv_4, deconvolution(conv_16,
                                               [batch_size, 128, 128, 32], name='deconv_3')], 3)
    conv_18 = convolution(conv_17, 32, name='conv_18')
    conv_19 = convolution(conv_18, 32, name='conv_19')
    
    # Sector 9
    conv_20 = tf.concat([conv_2, deconvolution(conv_19,
                                               [batch_size, 256, 256, 16], name='deconv_4')], 3)
    conv_21 = convolution(conv_20, 16, name='conv_21')
    conv_22 = convolution(conv_21, 16, name='conv_22')
    conv_23 = convolution(conv_22, 1, k_h=1, k_w=1,
                          activation=tf.nn.sigmoid)
    print ('conv_23 has shape: ', conv_23.shape)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv_23, labels=y))
    return conv_23, loss


# In[ ]:


def val(final):
    with tf.Session():
        final.eval()

def train(loss):
    generator, g_loss = build()
    with tf.variable_scope('optim', reuse=tf.AUTO_REUSE):
        t_vars = tf.trainable_variables()
        trainer = tf.train.AdamOptimizer(0.01).minimize(loss, var_list=t_vars)
    input_ = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                    dtype=np.uint8)
    output_ = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1),
                           dtype=np.uint8)
    for i in range(0, len(train_ids)):
        data = get_data(train_ids[i])
        input_[i] = data[0]
        output_[i] = data[1]
    wtf_x = input_[0:16,:,:,:]
    wtf_y = output_[0:16,:,:,:]
        
    with tf.Session() as sess:
        def mean_iou(y_pred,y_true):
            y_pred_ = tf.to_int64(y_pred > 0.5)
            y_true_ = tf.to_int64(y_true > 0.5)
            sess = tf.Session()
            score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
                return score.eval(session=sess)
        tf.global_variables_initializer().run()
        for ep_id in range(0, epoch):
            b_id = 0
            batches = int(ceil(float(len(train_ids))/float(batch_size)))
            for j in range(0, batches):
                wtf_x = input_[b_id*batch_size:(b_id+1)*batch_size,:,:,:]
                wtf_y = output_[b_id*batch_size:(b_id+1)*batch_size,:,:,:]
                if j%16 == 0:
                    s = randint(0,len(train_ids))
                    wtf_x = input_[s:s+16]#
                    wtf_y = output_[s:s+16]
                    sht = logit.eval({X:wtf_x, y:wtf_y})
                    print('validation: ', j/16 + 1, 'sht: ', tf.reduce_mean(sht).eval(), 'real: ', 
                          tf.reduce_mean(wtf_y).eval())
                _ = sess.run(trainer,
                             feed_dict={X: wtf_x,
                                        y: wtf_y}
                            )
                ans = loss.eval({X:wtf_x, y:wtf_y})
                
                b_id += 1
                print ('epoch: ', ep_id, ', loss: ', ans)


# In[ ]:


def mean_iou(y_pred,y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    sess = tf.Session()
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
        tf.global_variables_initializer().run(session=sess)
        return score.eval(session=sess)


# In[ ]:


logit, loss = build()
train(loss)

