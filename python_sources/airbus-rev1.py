#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd 
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


print(os.listdir("../input"))

DATA_PATH = '../input'
TEST_DATA = os.path.join(DATA_PATH, "test")
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(TRAIN_DATA, "masks")

df = pd.read_csv(DATA_PATH+'/train_ship_segmentations.csv')

path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values

df = df.set_index('ImageId')
df.head(5)


# In[ ]:


def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname, 0)
    cv2.imshow('img', img)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[ ]:


def canny_edge(data_frame):
    img_array = []
    len_test = [0,1]
    image_type = 'Train'
    for i in range(len(len_test)):#range(len(data_frame.index)):
        image_id = data_frame.index[i]
        print(image_id)
        get_image_data(image_id, image_type)
        img_canny = cv2.Canny(img_array[j], 0, 0)
        if j<2:
            cv2.imshow('original', image_id[j])
            cv2.imshow('canny', img_canny)
            j +=1


# In[ ]:


canny_edge(df)

