#!/usr/bin/env python
# coding: utf-8

# Training [kaggle kernal](https://www.kaggle.com/micheomaano/tpu-training-tensorflow-iafoos-method-42x256x256x3)  is available based on Tensorflow TPU to train 42x256x256x3 from intermediate layer in less than 3 hours. It can be improved alot.
# I hope you will like this work.
# Have Fun.

# In[ ]:


get_ipython().system('pip install ../input/kaggle-efficientnet-repo/efficientnet-1.0.0-py3-none-any.whl')

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import os
import skimage.io
from scipy.ndimage import measurements
import os
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from kaggle_datasets import KaggleDatasets
from tensorflow.keras import layers as L
import efficientnet.tfkeras as efn
from tensorflow.keras.utils import to_categorical
import gc
import albumentations
gc.enable()


# In[ ]:


sz = 256
N = 48

def tile(img):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:42]
    img = img[idxs]
    return img


# In[ ]:


class ConvNet(tf.keras.Model):

    def __init__(self, engine, input_shape, weights):
        super(ConvNet, self).__init__()
        
        self.engine = engine(
            include_top=False, input_shape=input_shape, weights=weights)
        
        
        self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.dense_2 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, (-1, IMG_SIZE, IMG_SIZE, 3))
        x = self.engine(x)
        shape = x.shape
        x = tf.reshape(x, (-1, N_TILES, shape[1], shape[2], shape[3])) 
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        x = tf.reshape(x, (-1, shape[1], N_TILES*shape[2], shape[3])) 
        x = self.avg_pool2d(x)
        x = self.dropout(x, training=False)
        x = self.dense_1(x)
        x = tf.nn.relu(x)
        return self.dense_2(x)


# In[ ]:


is_ef = True
backbone_name = 'efficientnet-b0'
N_TILES = 42
IMG_SIZE = 256


if backbone_name.startswith('efficientnet'):
    model_fn = getattr(efn, f'EfficientNetB{backbone_name[-1]}')
    
model = ConvNet(engine=model_fn, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights=None)
dummy_data = tf.zeros((2 * N_TILES, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
_ = model(dummy_data)


# In[ ]:


model.load_weights('../input/tpu-training-tensorflow-iafoos-method-42x256x256x3/efficientnet-b0.h5')


# In[ ]:


TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'
MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'
BASE_PATH = '../input/prostate-cancer-grade-assessment/'
train = pd.read_csv(BASE_PATH + "train.csv")
train.head()


# In[ ]:


sub = pd.read_csv("../input/prostate-cancer-grade-assessment/sample_submission.csv")
sub.head()


# In[ ]:


test = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv")
test.head()


# In[ ]:


TEST = '../input/prostate-cancer-grade-assessment/test_images/'


# In[ ]:


PRED_PATH = TEST
df = sub
t_df = test


# In[ ]:


transforms = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])


# In[ ]:


if os.path.exists(PRED_PATH):
    predictions = []
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        
        
        image_id = row['image_id']
        
        img_path = PRED_PATH + image_id + '.tiff' #BASE_PATH
        
        img = skimage.io.MultiImage(img_path)[1]
        
        patches = tile(img)
        patches1 = patches.copy()
        patches2 = patches.copy()
        k = 0
        while k < 42:
            patches1[k, ] = transforms(image=patches1[k, ])['image']
            patches2[k, ] = transforms(image=patches2[k, ])['image']
            k += 1
        
        image = np.stack([patches, patches1, patches2])
        image = image / 255.0
        
        pred = model.predict(image)
        isup = np.round(np.mean(pred))
        if isup < 0:
            isup = 0
        if isup > 5:
            isup = 5   
        predictions.append(int(isup))
        del patches, img
        gc.collect()


# In[ ]:


if os.path.exists(PRED_PATH):
    sub['isup_grade'] = predictions
    sub.to_csv("submission.csv", index=False)
else:
    sub.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




