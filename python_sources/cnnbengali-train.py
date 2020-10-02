#!/usr/bin/env python
# coding: utf-8

# This creates a model that takes up ~1GB memory, then loops through the parquet files to load the data and train the model on it. I am trying to free up memory after each loop, but it still accumulates and I end up exceeding the 13GB limit in the 3rd loop. Could someone provide some feedback on things I could be doing differently to stay under the limit?

# In[ ]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import gc
import cv2


# In[ ]:


#Credit to iafoss for the preprocessing code
HEIGHT = 137
WIDTH = 236
SIZE = 128

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


# In[ ]:


densenodes = 1000
convchannels = 50
kernsize = 3
poolsize = 6
batchsize = 100

model1 = tf.keras.Sequential([
tf.keras.layers.Conv2D(convchannels, (kernsize,kernsize), padding='valid', activation=tf.nn.relu, input_shape=(128,128,1)),
tf.keras.layers.MaxPooling2D((poolsize, poolsize), strides=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(densenodes, activation=tf.nn.relu),
tf.keras.layers.Dense(168,  activation=tf.nn.softmax)
])

model1.compile(optimizer=eval('tf.keras.optimizers.RMSprop(lr=0.001)'), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


for i in range(4):
    df = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_' + str(i) + '.parquet')
    trainimgs = df.iloc[:,0]
    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    resized = []
    for idx in range(len(df)):
        img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
        resized.append(crop_resize(img))
    del df
    del data
    gc.collect()
    train = np.array(resized).reshape(len(resized),128,128,1)
    del resized
    gc.collect()
    trainlbl = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv').merge(trainimgs).iloc[:,1]
    train_data_gen = ImageDataGenerator(rescale=1./255).flow(train,trainlbl)
    del train
    gc.collect()
    num_train_examples = len(trainimgs)
    model1.fit_generator(train_data_gen, epochs=10, steps_per_epoch=math.ceil(num_train_examples/batchsize), verbose=1)
    del train_data_gen
    gc.collect()


# In[ ]:




