#!/usr/bin/env python
# coding: utf-8

# - In this kernel, I will show you how to use the keras efficientNet for APTOS 2019, you can customise the rest
# 
# - orginal code:
# [https://github.com/qubvel/efficientnet](http://)
# 
# - original weights: [https://www.kaggle.com/kerneler/starter-efficientnet-keras-weights-b0-55f27ab0-b](http://)

# In[ ]:


import os
print(os.listdir("../input/efficientnet/efficientnet-master/efficientnet-master/efficientnet"))
import sys
sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))
from efficientnet import EfficientNetB5


# In[ ]:


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras import applications
from keras.layers import *
from keras.models import *
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# # EfficientNet

# In[ ]:


from efficientnet import EfficientNetB5

effnet = EfficientNetB5(
    weights= None, 
    include_top=False,
    input_shape=(224,224,3)
)

def build_model():
    model = Sequential()
    model.add(effnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0005),
        metrics=['accuracy']
    )
    
    return model
model = build_model()


# In[ ]:


model.summary()

