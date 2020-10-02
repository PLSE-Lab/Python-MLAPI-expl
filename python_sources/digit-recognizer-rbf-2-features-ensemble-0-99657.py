#!/usr/bin/env python
# coding: utf-8

# # Does RBF kernel (GaussianKernel) with two fixed Landmarks work well in ensemble?
# In the ensemble method, it is necessary that the correlation between multiple learner is low (or the independence of multiple learner is high). In order to obtain various classifiers with high independence, it is necessary to use a significantly different algorithm. Can we obtain a highly independent learner in a neural network with the same structure?  
# 
# In order to confront this issue, we challenged to obtain a highly independent learner by using RBF kernel with fixed landmarks in classifier just before output of the network.  
# 
# In this article we used keras. We used the [GaussianKernel layer](https://github.com/darecophoenixx/wordroid.sblo.jp/tree/master/lib/keras_ex/gkernel) to implement the RBF kernel.  

# ## Result
# |...|kaggle LB score|
# |:--|--:|
# |[10001](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10001)|0.77528|
# |[10002](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10002)|0.86085|
# |[10003](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10003)|0.83028|
# |[10004](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10004)|0.82985|
# |[10005](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10005)|0.85571|
# |[10006](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10006)|0.87328|
# |||
# |[ensemble007](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-2-ensemble007) (6 learners above)|**<span style="color:blue">0.99228</span>**|
# |[ensemble018](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-ensemble018) (90 learners)|**<span style="color:blue">0.99514</span>**|

# # Overview
# * Network in Handwritten Digit Recognizer
# * Independence of multiple learner
# * Result of the ensemble
# * Another network

# # Network in Handwritten Digit Recognizer

# ## Whole model
# |...|MODEL|INPUT|OUTPUT|DESC|
# |-------------|:-------------|-----:|-----:|:-----|
# |1| image converter | 28x28x1 | 128 |   |
# |2| classifier | 128 | 10 | quasi SVM(RBF) |

# ## 1 : image converter
# This model outputs feature values of images
# ```
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 784)          0                                            
# __________________________________________________________________________________________________
# reshape_1 (Reshape)             (None, 28, 28, 1)    0           input_1[0][0]                    
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 28, 28, 32)   832         reshape_1[0][0]                  
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 28, 28, 32)   25632       conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 14, 14, 32)   0           conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 14, 14, 64)   18496       max_pooling2d_1[0][0]            
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 14, 14, 64)   36928       conv2d_3[0][0]                   
# __________________________________________________________________________________________________
# global_max_pooling2d_1 (GlobalM (None, 64)           0           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# global_average_pooling2d_1 (Glo (None, 64)           0           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 128)          0           global_max_pooling2d_1[0][0]     
#                                                                  global_average_pooling2d_1[0][0] 
# __________________________________________________________________________________________________
# batch_normalization_1 (BatchNor (None, 128)          512         concatenate_1[0][0]              
# ==================================================================================================
# Total params: 82,400
# Trainable params: 82,144
# Non-trainable params: 256
# __________________________________________________________________________________________________
# ```

# ## 2 : classifier
# We construct quasi SVM using RBF kernel (Gaussian kernel) and Dense layer (see [Dense vs. GaussianKernel in moon data](https://github.com/darecophoenixx/wordroid.sblo.jp/wiki/%5BGaussianKernel-layer%5D-Dense-vs.-GaussianKernel-in-moon-data)). Two fixed landmarks are used for the RBF kernel (Gaussian kernel). By using different values for this fixed landmark, independence of each learner is guaranteed. This is empirical, not mathematically proven.
# 
# |...|LAYER|INPUT|OUTPUT|DESC|
# |:-:|:-------------|---:|---:|:-----|
# |1| GaussianKernel(RBF) | 128 | 2 |using TWO fixed Landmarks|
# |2| Dense | 2 | 10 | |

# # Independence of multiple learner
# Let's see the output of the RBF kernel in several learner using different landmarks. The output of this layer is two-dimensional since using two landmarks.
# 
# |state|image|state|image|
# |:-|:-:|:-|:-:|
# |[10001](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10001)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00105_s10001.png)|[10002](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10002)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00105_s10002.png)|
# |[10003](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10003)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00105_s10003.png)|[10004](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10004)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00105_s10004.png)|
# |[10005](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10005)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00105_s10005.png)|[10006](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10006)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00105_s10006.png)|
# state :: random state

# # Result of the ensemble
# The following is f1_score in the training data of each learner and Kaggle LB score in test data.  
# 
# |state|training data<br/>(f1 macro avg)|test data<br/>kaggle LB score|
# |:--|--:|--:|
# |[10001](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10001)|0.72|0.77528|
# |[10002](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10002)|0.89|0.86085|
# |[10003](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10003)|0.81|0.83028|
# |[10004](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10004)|0.84|0.82985|
# |[10005](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10005)|0.87|0.85571|
# |[10006](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-5-s10006)|0.89|0.87328|

# The results of ensemble by soft voting of these results are as follows.  
# 
# |state|training data<br/>(f1 macro avg)|test data<br/>kaggle LB score|
# |:--|--:|--:|
# |[ensemble007](https://www.kaggle.com/wordroid/digit28-only-2-features-gkernel-bn-2-ensemble007) (6 learners above)|0.99795|0.99228|
# |[ensemble018](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-ensemble018) (90 learners)|0.99905|0.99514|

# # Another network
# We changed the classifier in the network as follows.
# 
# |...|LAYER|INPUT|OUTPUT|DESC|
# |:-:|:-------------|---:|---:|:-----|
# |1| GaussianKernel(RBF) | 128 | 2 |using TWO fixed Landmarks|
# |2| <span style="color:blue">GaussianKernel(RBF)</span> | 2 | 10 | |

# ## Independence of multiple learner
# |state|image|state|image|
# |:-|:-:|:-|:-:|
# |[10001](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10001)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00106_s10001.png)|[10002](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10002)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00106_s10002.png)|
# |[10003](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10003)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00106_s10003.png)|[10004](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10004)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00106_s10004.png)|
# |[10005](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10005)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00106_s10005.png)|[10006](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10006)|![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/ttt00106_s10006.png)|
# state :: random state

# ## result
# |state|training data<br/>(f1 macro avg)|test data<br/>kaggle LB score|
# |:--|--:|--:|
# |[10001](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10001)|0.98|0.95700|
# |[10002](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10002)|1.00|0.97128|
# |[10003](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10003)|1.00|0.97714|
# |[10004](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10004)|0.99|0.97014|
# |[10005](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10005)|0.99|0.96742|
# |[10006](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-s10006)|0.98|0.95242|
# |[ensemble008](https://www.kaggle.com/wordroid/digit28-2-features-gkernel-bn-gkgk-ensemble008) (6 learners above)|0.99902|0.99200|

# The score of this kernel is the result of ensemble of 90 learners.
# 
# |state|training data<br/>(f1 macro avg)|test data<br/>kaggle LB score|
# |:--|--:|--:|
# |this kernel|0.99801|0.99657|

# 

# 

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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
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


# In[ ]:


# Load the data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


y_train = train['label'].values
print(y_train.shape)

y_cat = to_categorical(y_train)
y_cat.shape


# In[ ]:


x_train = train.iloc[:,1:].values / 255.0
print((x_train.min(), x_train.max()))
x_train.shape


# In[ ]:


x_train_fl = x_train.reshape((x_train.shape[0], -1))
x_train_fl.shape


# In[ ]:


plt.imshow(x_train[0].reshape((28,28)))


# In[ ]:


x_test = test.iloc[:,:].values / 255.0
print((x_test.min(), x_test.max()))
x_test.shape


# In[ ]:


x_test_fl = x_test.reshape((x_test.shape[0], -1))
x_test_fl.shape


# In[ ]:


plt.imshow(x_test[0].reshape((28,28)))


# In[ ]:


sample_submit = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
print(sample_submit.shape)
sample_submit.head()


# In[ ]:


def get_pred(src, rs=10001, show=False):
    print(src, rs)
    y_pred0 = pd.read_csv(os.path.join(src, 'proba_s{}.csv'.format(rs)))
    if show:
        print(y_pred0.shape)
        print(y_pred0.head())
    y_pred0_test = pd.read_csv(os.path.join(src, 'proba_test_s{}.csv'.format(rs)))
    if show:
        print(y_pred0_test.shape)
        print(y_pred0_test.head())
    
    return y_pred0.get_values(), y_pred0_test.get_values()


# ## -###-

# In[ ]:


y_pred_list = []
y_pred_test_list = []



src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s101zz'
rs_init = 10101
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s102zz'
rs_init = 10201
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s103zz'
rs_init = 10301
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s104zz'
rs_init = 10401
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s105zz'
rs_init = 10501
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s106zz'
rs_init = 10601
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s107zz'
rs_init = 10701
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s108zz'
rs_init = 10801
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)

src = '../input/digit28-2-features-gkernel-bn-da-gkgk-s109zz'
rs_init = 10901
for ii in range(10):
    rs = rs_init + ii
    y_pred_0, y_pred_test_0 = get_pred(src, rs)
    y_pred_list.append(y_pred_0)
    y_pred_test_list.append(y_pred_test_0)


# In[ ]:


len(y_pred_list)


# In[ ]:


y_pred = np.stack(y_pred_list)
print(y_pred.shape)

pred = y_pred.mean(axis=0)
print(pred.shape)
print(pred[0])


# In[ ]:


print(f1_score(y_train, np.argmax(pred, axis=1), average='macro'))
print(classification_report(y_train, np.argmax(pred, axis=1)))
confusion_matrix(y_train, np.argmax(pred, axis=1))


# In[ ]:


y_pred_test = np.stack(y_pred_test_list)
print(y_pred_test.shape)
pred_test = y_pred_test.mean(axis=0)
print(pred_test.shape)
print(pred_test[0])

submit_csv = sample_submit.copy()
submit_csv.Label = np.argmax(pred_test, axis=1)
submit_csv.head()
submit_csv.to_csv('submit.csv', index=False)


# In[ ]:





# In[ ]:




