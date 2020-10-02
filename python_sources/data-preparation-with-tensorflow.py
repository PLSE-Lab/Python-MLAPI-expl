#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random, datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle


# In[ ]:


TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 512
CHANNELS = 3
n_Y = 2


# In[ ]:


## tool functions
def ex_time(func):
    start_time = datetime.datetime.now()
    
    def wrapper(*args, **kwargs):
        print("start time: {}".format(start_time))
        res = func(*args, **kwargs)
        
        end_time = datetime.datetime.now()
        ex_time = end_time - start_time
        print("end time: {}".format(end_time))
        print("excute time: {} seconds".format(ex_time.seconds))

        return res
       
    return wrapper


# In[ ]:


filenames = os.listdir(TRAIN_DIR)
random.shuffle(filenames)


# In[ ]:


@ex_time
def load_images(filenames, dirname=TRAIN_DIR):
    file_name = tf.placeholder(dtype=tf.string)
    file = tf.read_file(file_name)
    img = tf.image.decode_jpeg(file)
    resized_img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    n_x = IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS
    m = len(filenames)
    
    images = np.zeros((m, n_x))
    labels = np.zeros((m, n_Y))
    with tf.Session() as sess:
        for i in range(m):
            filename = dirname + filenames[i]
            img = sess.run(resized_img, feed_dict={file_name: filename}).reshape(n_x, )   
            images[i] = img
            
            if "dog" in filename:
                labels[i, 1] = 1
            else:
                labels[i, 0] = 1
    images /= 255.0 
    return images.T, labels.T


# In[ ]:


images, labels = load_images(filenames[:1000])


# In[ ]:


def split_data(images, labels, dev_ratio=DEV_RATIO):
    dev_count = int(labels.shape[1] * DEV_RATIO)

    dev_images = images[:, :dev_count]
    train_images = images[:, dev_count:]
    dev_labels = labels[:, :dev_count]
    train_labels = labels[:, dev_count:]
    
    print("train images shape: {}, train labels shape:{},     dev images shape: {}, dev labels shape: {}".format(train_images.shape, train_labels.shape, dev_images.shape, dev_labels.shape))
    
    return train_images, train_labels, dev_images, dev_labels


# In[ ]:


train_images, train_labels, dev_images, dev_labels = split_data(images, labels)


# In[ ]:


print('Mean aspect ratio: ',np.mean(labels))
plt.plot(labels)
plt.show()


# In[ ]:




