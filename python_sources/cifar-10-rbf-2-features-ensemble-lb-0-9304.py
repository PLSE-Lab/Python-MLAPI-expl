#!/usr/bin/env python
# coding: utf-8

# [Cifar-10 RBF kernel 2 features ensemble [LB 0.9304]](https://www.kaggle.com/c/cifar-10/discussion/83473)

# In[ ]:


ls -la ../input


# In[ ]:


import datetime
now = datetime.datetime.now()
print(now)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import datetime
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
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score)
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D, GlobalAveragePooling2D,     GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
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
from keras.utils import to_categorical, plot_model
from keras import backend as K
import keras


# In[ ]:


from PIL import Image
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm
import datetime


# ### Load the data

# In[ ]:


src_dir = '../input/cifar10-object-recognition-in-images-zip-file'
train_zip = os.path.join(src_dir, 'train_test/train.zip')
test_zip = os.path.join(src_dir, 'train_test/test.zip')


# In[ ]:


train_labels = pd.read_csv(os.path.join(src_dir, "trainLabels.csv"))
print(train_labels.shape)
train_labels.head(10)


# In[ ]:


id_key = dict([ee for ee in enumerate(np.unique(train_labels.label.values))])
id_key


# In[ ]:


key_id = dict([(ee[1], ee[0]) for ee in enumerate(np.unique(train_labels.label.values))])
key_id


# In[ ]:


y_train = np.array([key_id[ee] for ee in train_labels.label.values])
y_train


# In[ ]:





# In[ ]:


def get_pred(src, show=False):
    print(src)
    y_pred0 = pd.read_csv(os.path.join(src, 'proba.csv'))
    print(y_pred0.shape)
    y_pred0_test = pd.read_csv(os.path.join(src, 'proba_test.csv'))
    print(y_pred0_test.shape)
    
    y_pred0.drop(['label'], axis=1, inplace=True)
    if show:
        print(y_pred0.head())
    try:
        y_pred0_test.drop(['label'], axis=1, inplace=True)
    except:
        pass
    if show:
        print(y_pred0_test.head())
    return y_pred0.get_values(), y_pred0_test.get_values()


# ## -###-

# In[ ]:


srcs = [
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10001',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10002',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10003',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10004',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10005',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10006',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10007',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10008',
    '../input/cifar10-gkernel-2-features-gkgk7-da-s10009',
]
y_pred_list = []
y_pred_test_list = []
for src in srcs:
    y_pred_0, y_pred_test_0 = get_pred(src)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)


# In[ ]:


y_pred = np.stack(y_pred_list)
print(y_pred.shape)

pred = y_pred.mean(axis=0)
print(pred.shape)
print(pred[0])


# In[ ]:


y_mat = np.c_[
    y_pred_list[0].flatten(),
    y_pred_list[1].flatten(),
    y_pred_list[2].flatten(),
]
y_mat.shape


# In[ ]:


cormat = np.corrcoef(y_mat.T)
cormat.shape


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (15.0, 12.0)
sns.heatmap(cormat, annot=True)


# In[ ]:





# In[ ]:


print(f1_score(y_train, np.argmax(pred, axis=1), average='macro'))
print(classification_report(y_train, np.argmax(pred, axis=1)))
confusion_matrix(y_train, np.argmax(pred, axis=1))


# In[ ]:


accuracy_score(y_train, np.argmax(pred, axis=1))


# In[ ]:


y_pred_test = np.stack(y_pred_test_list)
pred_test = y_pred_test.mean(axis=0)
print(pred_test.shape)
print(pred_test[0])
np.argmax(pred_test, axis=1)
# print(f1_score(y_test, np.argmax(pred_test, axis=1), average='macro'))
# print(classification_report(y_test, np.argmax(pred_test, axis=1)))
# confusion_matrix(y_test, np.argmax(pred_test, axis=1))


# In[ ]:


# accuracy_score(y_test, np.argmax(pred_test, axis=1))


# In[ ]:


src_dir = '../input/cifar10-object-recognition-in-images-zip-file'
test_labels = pd.read_csv(os.path.join(src_dir, "sampleSubmission.csv"))
print(test_labels.shape)
test_labels.head()


# In[ ]:


submit_csv = test_labels.copy()
submit_csv.label = [id_key[ee] for ee in np.argmax(pred_test, axis=1)]

submit_csv.to_csv('submit.csv', index=False)
submit_csv.head()


# In[ ]:





# In[ ]:





# In[ ]:




