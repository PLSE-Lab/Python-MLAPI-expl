#!/usr/bin/env python
# coding: utf-8

# **This kernel implements VGG on APTOS data **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


import gc
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##############################################################

import xgboost as xgb
import sklearn.ensemble as ensem
from keras.preprocessing.image import ImageDataGenerator

###############################################################

import sklearn.metrics as metrics
from sklearn.utils import shuffle
##############################################################


from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.utils import to_categorical
from keras.layers import ZeroPadding2D
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras import optimizers
from keras.applications import VGG16, ResNet50,Xception, InceptionResNetV2
from keras.applications import VGG19, InceptionV3,MobileNet
from keras.applications import DenseNet121, DenseNet169,DenseNet201
from keras.applications import NASNetLarge, NASNetMobile,MobileNetV2
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


import keras.optimizers as optim


# In[ ]:


colorcode = cv2.COLOR_BGR2RGB
interpolateVal = cv2.INTER_AREA
imgsize = 150
channle = 3


# In[ ]:


train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

paths = train.id_code
curr = paths[0]
images = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{curr}.png')
images = cv2.cvtColor(images, colorcode)
images = cv2.resize(images,(imgsize,imgsize),
                    interpolation=interpolateVal)
images = images.reshape((1,imgsize,imgsize,channle))
paths = paths[1:]
for path in tqdm(paths) :
    img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{path}.png')
    img = cv2.cvtColor(img, colorcode)
    img = cv2.resize(img,(imgsize,imgsize),
                     interpolation=interpolateVal)
    img = img.reshape((1,imgsize,imgsize,channle))
    images = np.vstack((images,img))
    
images = images/255


# In[ ]:


trainx = images
trainy = train.diagnosis


# In[ ]:


image_gen = ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 3],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last'
                              )


# In[ ]:


data0 = trainx[trainy ==0]
img0diag = np.repeat(0,data0.shape[0])


data1 = trainx[trainy ==1]
for imag in tqdm(data1) :
    it = image_gen.flow(imag.reshape((1,imgsize,imgsize,3)))
    for i in range(5) :
        data1 = np.append(data1,it.next().reshape((1,imgsize,imgsize,3)),axis=0)
img1diag = np.repeat(1,data1.shape[0])



data2 = trainx[trainy ==2]
for imag in tqdm(data2) :
    it = image_gen.flow(imag.reshape((1,imgsize,imgsize,3)))
    for i in range(1) :
        data2 = np.append(data2,it.next().reshape((1,
                                                   imgsize,imgsize,3)),
                          axis=0)
img2diag = np.repeat(2,data2.shape[0])


data3 = trainx[trainy ==3]
for imag in tqdm(data3) :
    it = image_gen.flow(imag.reshape((1,imgsize,imgsize,3)))
    for i in range(8) :
        data3 = np.append(data3,it.next().reshape((1,
                                                   imgsize,imgsize,3)),
                          axis=0)
img3diag = np.repeat(3,data3.shape[0])


data4 = trainx[trainy ==4]
for imag in tqdm(data4) :
    it = image_gen.flow(imag.reshape((1,imgsize,imgsize,3)))
    for i in range(5) :
        data4 = np.append(data4,it.next().reshape((1,
                                                   imgsize,imgsize,3)),
                          axis=0)
img4diag = np.repeat(4,data4.shape[0])


trainx = np.append(data0,data1,axis=0)
trainx = np.append(trainx,data2,axis=0)
trainx = np.append(trainx,data3,axis=0)
trainx = np.append(trainx,data4,axis=0)

trainy = np.append(img0diag,img1diag,axis=0)
trainy = np.append(trainy,img2diag,axis=0)
trainy = np.append(trainy,img3diag,axis=0)
trainy = np.append(trainy,img4diag,axis=0)


# In[ ]:


pathk='../input/keraspretrainedmodel/keras-pretrain-model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
modeli = InceptionV3(include_top=False,
                           weights=pathk,
                           input_tensor=None,
                           input_shape=(imgsize,imgsize, 3)
                          )


# In[ ]:


modeli.summary()


# In[ ]:


modeli.trainable = True
setTrainable = False
for layer in modeli.layers:
    if layer.name in ['conv2d_86','conv2d_94','batch_normalization_94']:
        setTrainable = True
    if setTrainable:
        layer.trainable = True
        setTrainable = False
    else:
        layer.trainable = False


# In[ ]:


model = Sequential()
model.add(modeli)
model.add(Flatten())
model.add(Dense(2096, activation='relu'))
model.add(Dense(1096, activation='relu'))
model.add(Dense(5, activation='softmax'))


# * conv2d_94
# * conv2d_86
# 

# In[ ]:


trainy = to_categorical(trainy , 
                        num_classes=None, 
                        dtype='float32')


# In[ ]:


import keras.optimizers as optim


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer=optim.RMSprop(lr=2e-6), 
              metrics=['accuracy'])
model.fit(trainx, trainy,
           epochs=70, batch_size=100, verbose=2)


# In[ ]:


pathst = test.id_code
currt = pathst[0]
imagest = cv2.imread(f'../input/aptos2019-blindness-detection/test_images/{currt}.png')
imagest = cv2.cvtColor(imagest, colorcode)
imagest = cv2.resize(imagest,(imgsize,imgsize),
                     interpolation=interpolateVal)
imagest = imagest.reshape((1,imgsize,imgsize,3))
pathst = pathst[1:]
for patht in pathst :
    imgt = cv2.imread(f'../input/aptos2019-blindness-detection/test_images/{patht}.png')
    imgt = cv2.cvtColor(imgt, colorcode)
    imgt = cv2.resize(imgt,(imgsize,imgsize),
                      interpolation=interpolateVal)
    imgt = imgt.reshape((1,imgsize,imgsize,3))
    imagest = np.vstack((imagest,imgt))
    
imagest = imagest/255


# In[ ]:


predValTemp  = model.predict(imagest)
predVal = predValTemp.argmax(axis=-1)
test['diagnosis'] = predVal
test.to_csv("submission.csv",index=False, header=True)

