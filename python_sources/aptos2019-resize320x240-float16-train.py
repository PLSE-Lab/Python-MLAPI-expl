#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import os.path
import itertools
from itertools import chain

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cluster, datasets, mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D,     GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model, Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


# In[ ]:


from PIL import Image
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm


# In[ ]:


src_dir = '../input'


# In[ ]:


train_labels = pd.read_csv(os.path.join(src_dir, "train.csv"))
print(train_labels.shape)
train_labels.head(10)


# In[ ]:


y_train = train_labels.diagnosis.values
y_train


# In[ ]:


y_cat_train = to_categorical(y_train)
y_cat_train


# In[ ]:


test_labels = pd.read_csv(os.path.join(src_dir, "sample_submission.csv"))
print(test_labels.shape)
test_labels.head()


# In[ ]:


#RESIZE = (640, 480)
#RESIZE = (560, 420)
RESIZE = (320, 240)
def get_arr0(Id, test=False):
    if test:
        tgt = 'test_images'
    else:
        tgt = 'train_images'
    with open(os.path.join(src_dir, tgt, Id+'.png'), 'rb') as fp:
        img = Image.open(fp)
        arr = (np.asarray(img) / 255.)
    arr = cv2.resize(arr, RESIZE)
    arr = arr.astype('float16')
    return arr

arr0 = get_arr0('0083ee8054ee')
print(arr0.shape)
plt.imshow(arr0.astype('float32'))


# In[ ]:


arr0 = get_arr0('006efc72b638', test=True)
print(arr0.shape)
plt.imshow(arr0.astype('float32'))


# In[ ]:


x_train_img_list = []
for id0 in tqdm(train_labels.id_code.tolist()):
    arr0 = get_arr0(id0)
    x_train_img_list.append(arr0)


# In[ ]:


x_train_img = np.stack(x_train_img_list)
del x_train_img_list
x_train_img.shape


# In[ ]:


np.savez_compressed('train_img_float16', x=x_train_img)


# In[ ]:


# x_test_img_list = []
# for id0 in tqdm(test_labels.id_code.tolist()):
#     arr0 = get_arr0(id0, test=True)
#     x_test_img_list.append(arr0)


# In[ ]:


# x_test_img = np.stack(x_test_img_list)
# x_test_img.shape


# In[ ]:


# np.savez_compressed('test_img_float16', x=x_test_img)


# In[ ]:




