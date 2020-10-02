#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom

from tqdm.notebook import tqdm


# In[ ]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train.head(70)


# In[ ]:


train.Patient[train.SmokingStatus == 'Ex-smoker']


# ## Image visualization

# In[ ]:


d = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/11.dcm')


# In[ ]:


plt.figure(figsize = (10,10))

plt.imshow(d.pixel_array)


# In[ ]:


train_data = {}
for p in train.Patient.values:
    train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')


# ## AutoEncoder

# ![](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png)

# Pipeline released in this kernel is very simple:
# 
# * train encoder and used one for translate image in vector
# * visulize results vectors used PCA decomposition algorithm

# In[ ]:


from tensorflow.keras.utils import Sequence
import cv2

class IGenerator(Sequence):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys=list(train_data.keys()), train_data=train_data, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.train_data = train_data
        self.batch_size = batch_size
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        x = []
        keys = np.random.choice(self.keys, size = self.batch_size)
        for k in keys:
            try:
                i = np.random.choice(self.train_data[k], size=1)[0]
                d = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')
                x.append(cv2.resize(d.pixel_array / 2**10, (512, 512)))
            except:
                print(k, i)
        x = np.array(x)
        x = np.expand_dims(x, axis=-1)
        return x, x


# In[ ]:


from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU
)

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Nadam

def get_encoder(shape=(512, 512, 1)):
    def res_block(x, n_features):
        _x = x
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    
        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = Add()([_x, x])
        return x
    
    inp = Input(shape=shape)
    
    # 512
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # 256
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    for _ in range(2):
        x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # 128
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    for _ in range(2):
        x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # 64
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    for _ in range(3):
        x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # 32
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    for _ in range(3):
        x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)    
    
    # 16
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    for _ in range(3):
        x = res_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x) 
    
    # 8
    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    return Model(inp, x)



def get_decoder(shape=(8, 8, 1)):
    inp = Input(shape=shape)

    # 8
    x = UpSampling2D((2, 2))(inp)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # 16
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # 32
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # 64
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # 128
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # 256
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    return Model(inp, x)


# In[ ]:


encoder = get_encoder((512, 512, 1))
decoder = get_decoder((8, 8, 1))


# In[ ]:


inp = Input((512, 512, 1))
e = encoder(inp)
d = decoder(e)
model = Model(inp, d)


# In[ ]:


model.compile(optimizer=Nadam(lr=2*1e-3, schedule_decay=1e-5), loss='mse')


# In[ ]:


model.summary()


# In[ ]:


model.fit_generator(IGenerator(), steps_per_epoch=500, epochs=5)


# In[ ]:


keys = [k for k in list(train_data.keys()) if k not in ['ID00011637202177653955184', 'ID00052637202186188008618']]


# In[ ]:


emb = {}
for k in tqdm(keys, total=len(keys)):
    x = []
    for i in train_data[k]:
        d = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')
        x.append(cv2.resize(d.pixel_array / 2**10, (512, 512)))
    x = np.expand_dims(x, axis=-1)
    emb[k] = encoder.predict(x)


# In[ ]:


encoder.save('encoder.h5')


# In[ ]:


del encoder, decoder, model


# In[ ]:


del IGenerator


# In[ ]:


import gc

gc.collect()


# In[ ]:


del x

gc.collect()


# ## Visualization PCA components

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


vec = []
for k in emb:
    vec.extend(emb[k][..., 0].reshape((len(emb[k]), 64)).tolist())
    
pca = PCA(n_components=3)
pca.fit(vec)

del vec
gc.collect()


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


emb3 = {}

for k in tqdm(emb):
    emb3[k] = pca.transform(emb[k][..., 0].reshape((len(emb[k]), 64)).tolist())


# In[ ]:


plt.figure(figsize=(10,10))

vec = []
for k in train.Patient.values:
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    vec.extend(emb3[k].tolist())
vec = np.array(vec)

plt.plot(vec[:, 0], vec[:, 1], '.', alpha=0.05, label='Male')


# In[ ]:


plt.figure(figsize=(10,10))

vec = []
for k in train.Patient[train.Sex == 'Male'].values[:100]:
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    vec.extend(emb3[k].tolist())

vec = np.array(vec)

plt.plot(vec[:, 0], vec[:, 1], '.', alpha=0.05, label='Male')


vec = []
for k in train.Patient[train.Sex == 'Female'].values[:100]:
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    vec.extend(emb3[k].tolist())

vec = np.array(vec)

plt.plot(vec[:, 0], vec[:, 1], '.', alpha=0.05, label='Female')

plt.legend()


# In[ ]:


plt.figure(figsize=(10,10))

vec = []
for k in train.Patient[train.SmokingStatus == 'Ex-smoker'].values[:100]:
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    vec.extend(emb3[k].tolist())

vec = np.array(vec)

plt.plot(vec[:, 0], vec[:, 1], '.', alpha=0.05, label='Ex-smoker')


vec = []
for k in train.Patient[train.SmokingStatus == 'Never smoked'].values[:100]:
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    vec.extend(emb3[k].tolist())

vec = np.array(vec)

plt.plot(vec[:, 0], vec[:, 1], '.', alpha=0.05, label='Never smoked')


vec = []
for k in train.Patient[train.SmokingStatus == 'Currently smokes'].values[:100]:
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    vec.extend(emb3[k].tolist())

vec = np.array(vec)

plt.plot(vec[:, 0], vec[:, 1], '.', alpha=0.05, label='Currently smokes')

plt.legend()


# In[ ]:


train['mc1'] = 0
train['sc1'] = 0


# In[ ]:


for k in tqdm(train.Patient.values):
    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:
        continue
    train.loc[train.Patient == k,'mc1'] = emb3[k][:, 0].max()
    train.loc[train.Patient == k,'sc1'] = emb3[k][:, 0].std()


# ## Depend 1 component stats with targets

# In[ ]:


plt.figure(figsize=(10, 10))

plt.plot(train.mc1, train.FVC, '.')


# In[ ]:


plt.figure(figsize=(10, 10))

plt.plot(train.mc1, train.Percent, '.')

