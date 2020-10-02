#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import os.path
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

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D,     GlobalMaxPooling1D, GlobalMaxPooling2D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from keras import backend as K


# In[ ]:


from PIL import Image


# In[ ]:


ls -la ../input/


# In[ ]:


src_dir = '../input'
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


y_train0 = np.array([key_id[ee] for ee in train_labels.label.values])
y_train0


# In[ ]:


test_labels = pd.read_csv(os.path.join(src_dir, "sampleSubmission.csv"))
print(test_labels.shape)
test_labels.head()


# In[ ]:


from zipfile import ZipFile


# In[ ]:


with ZipFile(train_zip, 'r') as myzip:
    print(myzip.infolist()[:10])
    print(len(myzip.namelist()))
    print(myzip.namelist()[:10])


# In[ ]:


trainImg_list = []
with ZipFile(train_zip, 'r') as myzip:
    for ii in train_labels.id.values:
        with myzip.open('train/'+str(ii)+'.png') as tgt:
            img = Image.open(tgt)
            img_array = np.asarray(img)
            trainImg_list.append(img_array)


# In[ ]:


x_train0 = np.stack(trainImg_list) / 255.0
x_train0.shape


# In[ ]:


testImg_list = []
with ZipFile(test_zip, 'r') as myzip:
    for ii in test_labels.id.values:
        with myzip.open('test/'+str(ii)+'.png') as tgt:
            img = Image.open(tgt)
            img_array = np.asarray(img)
            testImg_list.append(img_array)


# In[ ]:


x_test = np.stack(testImg_list) / 255.0
x_test.shape


# In[ ]:


plt.imshow(x_train0[0])


# In[ ]:


plt.imshow(x_test[0])


# In[ ]:


nrows=10
ncols=10
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = x_train0[ii*ncols + jj]
        iplt.imshow(img_array)


# In[ ]:


nrows=10
ncols=10
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = x_test[ii*ncols + jj]
        iplt.imshow(img_array)


# In[ ]:


nrows=10
ncols=12
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    idx = (y_train0 == ii)
    target_img = x_train0[idx][:ncols]
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = target_img[jj]
        iplt.imshow(img_array)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




