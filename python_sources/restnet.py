#!/usr/bin/env python
# coding: utf-8

# based on : 
# * https://www.kaggle.com/CVxTz/keras-cnn-starter
# * https://www.kaggle.com/jmourad100/keras-eda-and-cnn-starter
# * https://github.com/viig99/mkscancer
# 
# # || Loading Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os, time, random, cv2, glob, pickle, librosa
from pathlib import Path
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

from keras.models import Model
from keras.layers import (Convolution1D, Input, Dense, Flatten, Dropout, GlobalAveragePooling1D, concatenate,
                          Activation, MaxPool1D, GlobalMaxPool1D, BatchNormalization, Concatenate, ReLU, LeakyReLU)
from keras.layers import BatchNormalization, Activation, Conv1D, Concatenate, AveragePooling1D
from keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalAveragePooling1D

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import sparse_categorical_crossentropy
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers as ll

print(os.listdir("../input"))


# In[2]:


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    x = ll.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, strides=strides,
               name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    shortcut = BatchNormalization(name=bn_name_base + '1')(input_tensor)
    shortcut = Conv1D(filters3, 1, strides=strides,
                      name=conv_name_base + '1')(shortcut)

    x = ll.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_block(input_tensor, final_layer_output=128, append='n'):
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1' + append)(input_tensor)
    x = BatchNormalization(name='bn_conv1' + append)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)
    x = conv_block(x, 3, [64, 64, 256],
                   stage=2, block='a' + append, strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b' + append)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c' + append)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d' + append)
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a' + append)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + append)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c' + append)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d' + append)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e' + append)
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f' + append)
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a' + append)
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b' + append)
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c' + append)
    x = AveragePooling1D(final_layer_output, name='avg_pool' + append)(x)
    x = Flatten()(x)
    return x


# # || Configuration

# In[3]:


t_start = time.time()

# Keras reproduce score (then init all model seed)
seed_nb=14
import numpy as np 
np.random.seed(seed_nb)
import tensorflow as tf
tf.set_random_seed(seed_nb)


# # || Data Preparation

# In[4]:


input_length = 5000

batch_size = 128

def audio_norm(data):

    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5


def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000
    if len(data)>input_length:
        max_offset = len(data)-input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
        
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
            
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        
    data = audio_norm(data)
    return data


# In[5]:


train_files = glob.glob("../input/train_curated/*.wav")
train_labels = pd.read_csv("../input/train_curated.csv")
train_labels['labels'] = train_labels['labels'].apply(lambda x: x.split(',')[0]) # only keep first label for now


# In[6]:


file_to_label = {"../input/train_curated/"+k:v for k,v in zip(train_labels.fname.values, train_labels.labels.values)}


# In[7]:


list_labels = sorted(list(set(train_labels.labels.values)))
label_to_int = {k:v for v,k in enumerate(list_labels)}
int_to_label = {v:k for k,v in label_to_int.items()}
file_to_int = {k:label_to_int[v] for k,v in file_to_label.items()}


# In[8]:


def get_model():
    nclass = len(list_labels)
    model_input = Input(shape=(input_length, 1))
    output = resnet_block(model_input)    
    dense_1 = Dense(nclass, activation="softmax")(output)

    model = Model(inputs=model_input, outputs=dense_1)
    model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=['acc'])
    return model


# In[9]:


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[10]:


def train_generator(list_files, batch_size=batch_size):
    while True:
        random.shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:,:,np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)
            
            yield batch_data, batch_labels


# In[11]:


tr_files, val_files = train_test_split(train_files, test_size=0.05)


# In[12]:


model = get_model()


# In[13]:


# model.summary()


# In[16]:


model.fit_generator(train_generator(tr_files), 
                    steps_per_epoch=len(tr_files)//batch_size, 
                    validation_data=train_generator(val_files),
                    validation_steps=len(val_files)//batch_size,
                    epochs=2)


# In[ ]:


list_preds = []
batch_size = 128
test_files = glob.glob("../input/test/*.wav")
test_files.sort()


# In[ ]:


for batch_files in tqdm(chunker(test_files, size=batch_size), total=len(test_files)//batch_size ):
    batch_data = [load_audio_file(fpath) for fpath in batch_files]
    batch_data = np.array(batch_data)[:,:,np.newaxis]
    preds = model.predict(batch_data).tolist()
    list_preds += preds


# In[ ]:


array_preds = np.array(list_preds)


# In[ ]:


df = pd.read_csv('../input/sample_submission.csv')
for i, v in enumerate(list_labels):
    df[v] = array_preds[:, i]


# In[ ]:


df['fname'] = df.fname.apply(lambda x: x.split("/")[-1])


# In[ ]:


df.to_csv("submission.csv", index=False)
df.head()


# In[ ]:


t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")


# In[ ]:




