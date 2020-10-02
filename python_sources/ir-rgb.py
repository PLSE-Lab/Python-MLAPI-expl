#!/usr/bin/env python
# coding: utf-8

# # How much information can be extracted from the IR channel alone? 
# 
# Is this notebook we try to rebuild the RGB image purely from its IR channel with a U-convolutional architecture, and shortcut connections.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io, transform
from keras.preprocessing import image as kimage
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


TDATA_PATH = "../input/train-tif/"
JDATA_PATH = "../input/train-jpg/"


# In[ ]:


files = os.listdir(JDATA_PATH) + os.listdir(TDATA_PATH)
files = sorted(set([f.split(".")[0] for f in files]))


# In[ ]:


def open_tif_file(path, size=32):
    path = path + ".tif"
    img = io.imread(path)
    img = transform.resize(img, (32, 32))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float16)
    return img[:, :, :, [3]] # Only IR channel


def open_jpg_file(path, size=32):
    path = path + ".jpg"
    img = kimage.load_img(path, target_size=(size, size))
    img = kimage.img_to_array(img, data_format="channels_last")
    img = np.expand_dims(img, 0)
    img = img / 255.
    img = img.astype(np.float16)
    return img


def load_data(TDATA_PATH, JDATA_PATH, files, img_size,
              validation_size=2000, max_size=None):
    validation_indices = np.arange(len(files))

    np.random.seed(0)
    np.random.shuffle(validation_indices)
    validation_indices = validation_indices[:validation_size]

    xtr = []
    xva = []
    ytr = []
    yva = []

    for ith, file in enumerate(files):
        if (ith + 1) % 2500 == 0:
            print("Files loaded:", ith + 1)
            
        if max_size is not None and ith >= max_size:
            break

        try:
            x = open_tif_file(TDATA_PATH + file, img_size)
            y = open_jpg_file(JDATA_PATH + file, img_size)
        except Exception as e:
            print(e, ith)
        else:
            if ith in validation_indices:
                xva.append(x)
                yva.append(y)
            else:
                xtr.append(x)
                ytr.append(y)

    xtr = np.vstack(xtr)
    xva = np.vstack(xva)
    ytr = np.vstack(ytr)
    yva = np.vstack(yva)
    
    return xtr, ytr, xva, yva


# In[ ]:


IMG_SIZE = 32
VALIDATION_SIZE = 4000
MAX_SIZE = 2500


# In[ ]:


xtr, ytr, xva, yva = load_data(
    TDATA_PATH, JDATA_PATH, files, img_size=IMG_SIZE,
    validation_size=VALIDATION_SIZE, max_size=MAX_SIZE)


# In[ ]:


print(xtr.shape, ytr.shape)
print(xva.shape, yva.shape)


# In[ ]:


import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, AveragePooling2D
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Conv2DTranspose, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam


# In[ ]:


img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

d1 = Conv2D(50, (3, 3), padding="same", activation="relu", name='dconv1')(img_input)
x = MaxPooling2D(2)(d1)

d2 = Conv2D(50, (3, 3), padding="same", activation="relu", name='dconv2')(x)
x = MaxPooling2D(2)(d2)

d3 = Conv2D(50, (3, 3), padding="same", activation="relu", name='dconv3')(x)
x = MaxPooling2D(2)(d3)

x = Conv2D(50, (3, 3), padding="same", activation="relu", name='conv')(x)

x = UpSampling2D(2)(x)
x = add([x, d3])
u3 = Conv2D(50, (3, 3), padding="same", activation="relu", name='uconv3')(x)

x = UpSampling2D(2)(u3)
x = add([x, d2])
u2 = Conv2D(50, (3, 3), padding="same", activation="relu", name='uconv2')(x)

x = UpSampling2D(2)(u2)
x = add([x, d1])
x = Conv2D(50, (3, 3), padding="same", activation="relu", name='uconv11')(x)

x = add([x, img_input])
x = Conv2D(50, (2, 2), padding="same", activation="relu", name='uconv12')(x)
out = Conv2D(3, (1, 1), padding="same", activation="sigmoid", name="img_output")(x)

model = Model(inputs=img_input, outputs=out, name="u-conv")


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss="mse", optimizer=Adam(lr=5e-4))


# In[ ]:


# Loop over this cell multiple times to avoid time-out
h = model.fit(xtr, ytr, validation_data=[xva, yva], verbose=2, epochs=10)


# In[ ]:


def ir_channel(x):
    irx = np.tile(x, 3).astype(float)
    irx = irx - irx.min()
    irx = irx / irx.max()
    return irx


# Below we inspect if the model has converged yet by analyzing what sort of image it generates from training data

# In[ ]:


gentr = model.predict(xtr)
fig, axis = plt.subplots(4, 3 * 2, figsize=(15, 10))

for i in range(4):
    for j in range(0, 6, 3):
        axis[i][j + 0].imshow(ir_channel(xtr[i*2 + j//3]) )
        axis[i][j + 1].imshow(gentr[i*2 + j//3])
        axis[i][j + 2].imshow(ytr[i*2 + j//3].astype(float))
        
        axis[i][j].axis("off")
        axis[i][j + 1].axis("off")
        axis[i][j + 2].axis("off")


# Here we inspect what sort of RGB image is generated for validation samples.

# In[ ]:


gen = model.predict(xva)
fig, axis = plt.subplots(4, 3 * 2, figsize=(15, 10))

for i in range(4):
    for j in range(0, 6, 3):
        axis[i][j + 0].imshow( ir_channel(xva[i*2 + j//3]) )
        axis[i][j + 1].imshow(gen[i*2 + j//3])
        axis[i][j + 2].imshow(yva[i*2 + j//3].astype(float))
        
        axis[i][j].axis("off")
        axis[i][j + 1].axis("off")
        axis[i][j + 2].axis("off")
        


# Looks promising, however it needs to run longer and with more data.

# In[ ]:




