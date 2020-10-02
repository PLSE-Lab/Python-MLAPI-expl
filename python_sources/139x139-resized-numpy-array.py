#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import os
import shutil
print(os.listdir("../input"))


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

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D,     GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
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
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm


# In[ ]:


src_dir = '../input/human-protein-atlas-image-classification'


# In[ ]:


train_labels = pd.read_csv(os.path.join(src_dir, "train.csv"))
print(train_labels.shape)
train_labels.head(10)


# In[ ]:


test_labels = pd.read_csv(os.path.join(src_dir, "sample_submission.csv"))
print(test_labels.shape)
test_labels.head()


# In[ ]:


def show_arr(arr, nrows = 1, ncols = 4, figsize=(15, 5)):
    fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ii in range(ncols):
        iplt = subs[ii]
        img_array = arr[:,:,ii]
        if ii == 0:
            cp = 'Greens'
        elif ii == 1:
            cp = 'Blues'
        elif ii == 2:
            cp = 'Reds'
        else:
            cp = 'Oranges'
        iplt.imshow(img_array, cmap=cp)


# In[ ]:


def get_arr0(Id, test=False):
    def fn(Id, color, test=False):
        if test:
            tgt = 'test'
        else:
            tgt = 'train'
        with open(os.path.join(src_dir, tgt, Id+'_{}.png'.format(color)), 'rb') as fp:
            img = Image.open(fp)
            arr = (np.asarray(img) / 255.)
        return arr
    res = []
    for icolor in ['green', 'blue', 'red', 'yellow']:
        arr0 = fn(Id, icolor, test)
        res.append(arr0)
    arr = np.stack(res, axis=-1)
    return arr


# In[ ]:


arr = get_arr0('00008af0-bad0-11e8-b2b8-ac1f6b6435d0', test=True)
print(arr.shape)
show_arr(arr)


# In[ ]:


arr = get_arr0('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0')
print(arr.shape)
show_arr(arr)


# ### cache train data

# In[ ]:


SH = (139, 139)


# In[ ]:


### CACHE
img_cache_train = np.zeros((train_labels.shape[0], 139, 139, 4), dtype=np.float16)
ID_LIST_TRAIN = train_labels.Id.tolist()


# In[ ]:


for ii, id0 in enumerate(tqdm(ID_LIST_TRAIN)):
    arr = get_arr0(id0)
    img_cache_train[ii] = cv2.resize(arr[:], SH)


# In[ ]:


img_cache_train.shape


# In[ ]:


np.savez_compressed('img_cache_train_resize_139x139', x=img_cache_train)


# In[ ]:


ls -la


# In[ ]:





# In[ ]:





# In[ ]:




