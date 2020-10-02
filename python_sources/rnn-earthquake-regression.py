#!/usr/bin/env python
# coding: utf-8

# **Basic Information**
# 
# In this notebook, I build a Neural Network architecture to predict TTF.
# 
# The goal of this competition is to use seismic signals to predict the timing of laboratory earthquakes. The data comes from a well-known experimental set-up used to study earthquake physics. The acoustic_data input signal is used to predict the time remaining before the next laboratory earthquake (time_to_failure).
# 
# The training data is a single, continuous segment of experimental data. The test data consists of a folder containing many small segments. The data within each test file is continuous, but the test files do not represent a continuous segment of the experiment; thus, the predictions cannot be assumed to follow the same regular pattern seen in the training file.
# 
# For each seg_id in the test folder, you should predict a single time_to_failure corresponding to the time between the last row of the segment and the next laboratory earthquake.
# 
# * train length: 629,145,480
# * Max time_to_failure = 16.1074
# * Min time_to_failure = 9.5503965e-05
# * test length: 2624 * 150,000 = 393,600,000
# <br>
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import feather
import os
import gc
import multiprocessing
from tqdm import tqdm
from numba import jit
from keras import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.layers import Conv1D, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input,Concatenate,Reshape,CuDNNLSTM,CuDNNGRU,GlobalMaxPooling1D
from keras.layers import PReLU, LeakyReLU
from keras.optimizers import adam, rmsprop
from keras.regularizers import l1,l2, l1_l2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from scipy.stats import *
from sklearn.metrics import mean_absolute_error

#from scipy import rfft
from numpy.fft import *

import warnings
warnings.filterwarnings('ignore')

from numpy.random import seed
seed(1337)
from tensorflow import set_random_seed
set_random_seed(1337)


# In[ ]:


print(os.listdir('../input/LANL-Earthquake-Prediction/'))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = feather.read_dataframe('../input/lanl-ft/train.ft')\n# zero center \ntrain['acoustic_data'] = train['acoustic_data'].values - 4")


# In[ ]:


# to plot training/validation history object
def plt_dynamic(x, vy, ty, ax, colors=['b'], title=''):
    ax.plot(x, vy, 'b', label='Validation Loss')
    ax.plot(x, ty, 'r', label='Train Loss')
    plt.legend()
    plt.grid()
    plt.title(title)
    fig.canvas.draw()
    plt.show()


# In[ ]:


# get the earthquake indices. this is where the experiement resets
diff = np.diff(train['time_to_failure'].values)
end = np.nonzero(diff>0)[0]
start = end + 1
start = np.insert(start, 0, 0)
del diff
gc.collect()
start


# In[ ]:


def gen_batches_front(col, interval=150_000):
    high = []
    low = []
    splits =[]
    high_ttf = list(range(9))
    
    for i, beg in enumerate(start):
        counter = 0
        if beg != 621_985_673:
            last = start[i+1]
        else:
            last = len(train)
        last = (last-beg)//150_000 * 150_000 + beg
        
        for x in range(beg, last, interval):
            if col == 'acoustic_data':
                if i in high_ttf:
                    high.append(train[col].iloc[x:150_000+x].values)
                else:
                    low.append(train[col].iloc[x:150_000+x].values)
            else:
                if i in high_ttf:
                    high.append(train[col].iloc[x:150_000+x].values[-1])
                else:
                    low.append(train[col].iloc[x:150_000+x].values[-1])
                    
        # oversample the end points
        sample = 15000
        seg = 150000
        for z, y in enumerate(range(0, sample*11, sample)):
            if col == 'acoustic_data':
                if i in high_ttf:
                    high.append(train[col].iloc[last-seg*(z+1):last-seg*z].values)
                else:
                    low.append(train[col].iloc[last-seg*(z+1):last-seg*z].values)
            else:
                if i in high_ttf:
                    high.append(train[col].iloc[last-seg*(z+1):last-seg*z].values[-1])
                else:
                    low.append(train[col].iloc[last-seg*(z+1):last-seg*z].values[-1])
    return np.asarray(high), np.asarray(low)


# In[ ]:


# can modify the interval to a factor of 150000 for increased sampling
def preprocess_front():
    xtrain, xtest = gen_batches_front('acoustic_data', interval=150000)
    ytrain, ytest = gen_batches_front('time_to_failure', interval=150000)
    xtrain = xtrain.reshape(-1, 150000, 1)
    xtest = xtest.reshape(-1, 150000, 1)
    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)
    return xtrain, xtest, ytrain, ytest


# In[ ]:


xtrain, xtest, ytrain, ytest = preprocess_front()


# In[ ]:


gc.collect()
del train


# In[ ]:


checkpoint1 = ModelCheckpoint('best1.hdf5', verbose=0, save_best_only=True, mode='min')


# In[ ]:


get_ipython().run_cell_magic('time', '', "epochs=20\nmdl = Sequential()\nmdl.add(SeparableConv1D(32, 8, activation='relu', input_shape=(xtrain.shape[1],1)))\nmdl.add(CuDNNGRU(32, return_sequences=True))\nmdl.add(GlobalAveragePooling1D())\nmdl.add(Dense(32, activation='relu'))\nmdl.add(Dense(1))\nmdl.compile(loss='mae', optimizer=adam(lr=0.001))\nmdl.summary()")


# In[ ]:


#visualize network architecture
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(mdl).create(prog='dot', format='svg'))


# In[ ]:


history = mdl.fit(xtrain, ytrain, epochs=epochs, batch_size=64, verbose=2, 
                  validation_data=[xtest,ytest], callbacks=[checkpoint1])
fig, ax = plt.subplots(1,1)
vy = history.history['val_loss']
ty = history.history['loss']
ax.set_xlabel('Epoch')
x = list(range(1,epochs+1))
ax.set_ylabel('Mean Absolute Error')
plt_dynamic(x,vy,ty,ax, title='history')


# In[ ]:


x_train = np.concatenate([xtrain, xtest], axis=0)
y_train = np.concatenate([ytrain, ytest], axis=0)
oof_pred = np.zeros(len(x_train))
best_oof = np.zeros(len(x_train))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_pred = mdl.predict(xtrain)\nvalid_pred = mdl.predict(xtest)\noof_pred[xtrain.shape[0]:] += valid_pred[:,0]\nprint(f'Train MAE: {mean_absolute_error(ytrain, train_pred):.4f}')\nprint(f'Valid MAE: {mean_absolute_error(ytest, valid_pred):.4f}')")


# In[ ]:


plt.title('predicted train')
sns.distplot(train_pred)


# In[ ]:


plt.title('predicted test')
sns.distplot(valid_pred)


# In[ ]:


#compute absolute error
train_error = np.abs(np.subtract(train_pred.reshape(1,-1)[0], ytrain))
valid_error = np.abs(np.subtract(valid_pred.reshape(1,-1)[0], ytest))


# In[ ]:


#plot the train error distribution
plt.title('train error')
plt.xlabel('absolute error')
sns.distplot(train_error)


# In[ ]:


#plot the validation error distribution
plt.title('validation error')
plt.xlabel('absolute error')
sns.distplot(valid_error)


# In[ ]:


print(f'max ytrain: {np.max(ytrain):.4f}')
print(f'max ytest: {np.max(ytest):.4f}')
print(f'max p_train: {np.max(train_pred):.4f}')
print(f'max p_test: {np.max(valid_pred):.4f}')


# In[ ]:


#plot scatter
plt.figure(figsize=(9,9))
plt.scatter(ytrain, train_pred)
plt.xlabel('ytrue')
plt.ylabel('ypred')
plt.show()


# In[ ]:


#plot scatter
plt.figure(figsize=(9,9))
plt.scatter(ytest, valid_pred)
plt.title('oof predictions')
plt.xlabel('ytrue')
plt.ylabel('ypred')
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', "best1 = load_model('best1.hdf5')\ntrain_pred = best1.predict(xtrain)\nvalid_pred = best1.predict(xtest)\nbest_oof[xtrain.shape[0]:] += valid_pred[:,0]\nprint(f'Train MAE: {mean_absolute_error(ytrain, train_pred):.4f}')\nprint(f'Valid MAE: {mean_absolute_error(ytest, valid_pred):.4f}')")


# In[ ]:


plt.title('predicted train')
sns.distplot(train_pred)


# In[ ]:


plt.title('predicted test')
sns.distplot(valid_pred)


# In[ ]:


#compute absolute error
train_error = np.abs(np.subtract(train_pred.reshape(1,-1)[0], ytrain))
valid_error = np.abs(np.subtract(valid_pred.reshape(1,-1)[0], ytest))


# In[ ]:


#plot the train error distribution
plt.title('train error')
plt.xlabel('absolute error')
sns.distplot(train_error)


# In[ ]:


#plot the validation error distribution
plt.title('validation error')
plt.xlabel('absolute error')
sns.distplot(valid_error)


# In[ ]:


print(f'max ytrain: {np.max(ytrain):.4f}')
print(f'max ytest: {np.max(ytest):.4f}')
print(f'max p_train: {np.max(train_pred):.4f}')
print(f'max p_test: {np.max(valid_pred):.4f}')


# In[ ]:


del train_pred, valid_pred, train_error, valid_error
gc.collect()


# In[ ]:


checkpoint2 = ModelCheckpoint('best2.hdf5', verbose=0, save_best_only=True, mode='min')


# In[ ]:


get_ipython().run_cell_magic('time', '', "epochs=20\nmdl2 = Sequential()\nmdl2.add(SeparableConv1D(32, 8, activation='relu', input_shape=(xtrain.shape[1],1)))\nmdl2.add(CuDNNGRU(32, return_sequences=True))\nmdl2.add(GlobalAveragePooling1D())\nmdl2.add(Dense(32, activation='relu'))\nmdl2.add(Dense(1))\nmdl2.compile(loss='mae', optimizer=adam(lr=0.001))\nhistory = mdl2.fit(xtest, ytest, epochs=epochs, batch_size=64, verbose=2, \n                  validation_data=[xtrain, ytrain], callbacks=[checkpoint2])\nfig, ax = plt.subplots(1,1)\nvy = history.history['val_loss'][4:]\nty = history.history['loss'][4:]\nax.set_xlabel('Epoch')\nx = list(range(5,epochs+1))\nax.set_ylabel('Mean Absolute Error')\nplt_dynamic(x,vy,ty,ax, title='history')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_pred = mdl2.predict(xtrain)\nvalid_pred = mdl2.predict(xtest)\noof_pred[:xtrain.shape[0]] += train_pred[:,0]\nprint(f'Train MAE: {mean_absolute_error(ytrain, train_pred):.4f}')\nprint(f'Valid MAE: {mean_absolute_error(ytest, valid_pred):.4f}')")


# In[ ]:


#plot the train distibution
plt.title('predicted train')
sns.distplot(train_pred)


# In[ ]:


#plot the test distribution
plt.title('predicted test')
sns.distplot(valid_pred)


# In[ ]:


#compute absolute error
train_error = np.abs(np.subtract(train_pred.reshape(1,-1)[0], ytrain))
valid_error = np.abs(np.subtract(valid_pred.reshape(1,-1)[0], ytest))


# In[ ]:


#plot the train error distribution
plt.title('train error')
plt.xlabel('absolute error')
sns.distplot(train_error)


# In[ ]:


#plot the test error distribution
plt.title('validation error')
plt.xlabel('absolute error')
sns.distplot(valid_error)


# In[ ]:


print(f'max ytrain: {np.max(ytrain):.4f}')
print(f'max ytest: {np.max(ytest):.4f}')
print(f'max p_train: {np.max(train_pred):.4f}')
print(f'max p_test: {np.max(valid_pred):.4f}')


# In[ ]:


get_ipython().run_cell_magic('time', '', "best2 = load_model('best2.hdf5')\ntrain_pred = best2.predict(xtrain)\nvalid_pred = best2.predict(xtest)\nbest_oof[:xtrain.shape[0]] += train_pred[:,0]\nprint(f'Train MAE: {mean_absolute_error(ytrain, train_pred):.4f}')\nprint(f'Valid MAE: {mean_absolute_error(ytest, valid_pred):.4f}')")


# In[ ]:


plt.title('predicted train')
sns.distplot(train_pred)


# In[ ]:


plt.title('predicted test')
sns.distplot(valid_pred)


# In[ ]:


#plot scatter - visualize actual vs prediction
plt.figure(figsize=(9,9))
plt.scatter(ytrain, train_pred)
plt.title('oof predictions')
plt.xlabel('ytrue')
plt.ylabel('ypred')
plt.show()


# In[ ]:


#plot scatter - visualize actual vs prediction
plt.figure(figsize=(9,9))
plt.scatter(ytest, valid_pred)
plt.xlabel('ytrue')
plt.ylabel('ypred')
plt.show()


# In[ ]:


#compute absolute error
train_error = np.abs(np.subtract(train_pred.reshape(1,-1)[0], ytrain))
valid_error = np.abs(np.subtract(valid_pred.reshape(1,-1)[0], ytest))


# In[ ]:


#plot the train error distribution
plt.title('train error')
plt.xlabel('absolute error')
sns.distplot(train_error)


# In[ ]:


#plot the validation error distribution
plt.title('validation error')
plt.xlabel('absolute error')
sns.distplot(valid_error)


# In[ ]:


print(f'max ytrain: {np.max(ytrain):.4f}')
print(f'max ytest: {np.max(ytest):.4f}')
print(f'max p_train: {np.max(train_pred):.4f}')
print(f'max p_test: {np.max(valid_pred):.4f}')


# In[ ]:


del train_pred, valid_pred, train_error, valid_error
gc.collect()


# # OOF CV Results

# In[ ]:


print('overtrain mae: {:.4f}'.format(mean_absolute_error(y_train, oof_pred)))
print('best mae: {:.4f}'.format(mean_absolute_error(y_train, best_oof)))


# In[ ]:


plt.title('oof_pred')
sns.distplot(oof_pred)
plt.show()


# In[ ]:


plt.title('best_oof')
sns.distplot(best_oof)
plt.show()


# In[ ]:


plt.figure(figsize=(9,9))
plt.scatter(y_train, oof_pred)
plt.title('oof_pred')
plt.xlabel('ytrue')
plt.ylabel('ypred')
plt.show()


# In[ ]:


plt.figure(figsize=(9,9))
plt.scatter(y_train, best_oof)
plt.title('best_oof')
plt.xlabel('ytrue')
plt.ylabel('ypred')
plt.show()


# # Submission File
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "sub = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', \n                  dtype={'seg_id': 'category', 'time_to_failure':np.float32})\n\ntest_data = []\nfor fname in sub['seg_id'].values:\n    test_data.append(pd.read_csv('../input/LANL-Earthquake-Prediction/test/'+fname+'.csv', \n                                 dtype={'acoustic_data':np.int16})['acoustic_data'].values)\n# zero center & reshape\ntest_data = np.asarray(test_data) - 4\ntest_data = test_data.reshape(-1, 150_000, 1)")


# In[ ]:


#%time 
pred1 = best1.predict(test_data)
sub['time_to_failure'] = pred1
sub.to_csv('submission1.csv', index=False)
sub.head()


# In[ ]:


plt.title('first half')
sns.distplot(sub['time_to_failure'].values)


# In[ ]:


get_ipython().run_line_magic('time', 'pred2 = best2.predict(test_data)')
sub['time_to_failure'] = pred2
sub.to_csv('submission2.csv', index=False)
sub.head()


# In[ ]:


plt.title('second half')
sns.distplot(sub['time_to_failure'].values)


# In[ ]:


# blended first + second half
first = pd.read_csv('submission1.csv')
second = pd.read_csv('submission2.csv')
blend = first.copy()
blend['time_to_failure'] = (blend['time_to_failure'] + second['time_to_failure'])/2
blend.to_csv('frontback.csv', index=False)
blend.head()


# In[ ]:


plt.title('blended')
sns.distplot(blend['time_to_failure'].values)

