#!/usr/bin/env python
# coding: utf-8

# # Outputs features of signals divided in parts (e.g. 200 parts of 4000 steps) for both train and test set in +- five minutes.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse
from tqdm import tqdm_notebook as tqdm
from numba import jit
from matplotlib import pyplot as plt
from scipy import signal as scipy_signal
import gc
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


N_TRAIN = 8712  # nr of samples in training set
N_TEST = 20337
# N_TRAIN = 210  # to try out, also comment out save_data(1) below
P = 3  # nr of phases in signal
ADDRESS_TRAIN = '../input/train.parquet'
ADDRESS_TEST = '../input/test.parquet'
SPLITS_TRAIN = 3  # signal splits have to be divisable by 3 for align()
SPLITS_TEST = 3
LENGTH = 800000
N_PARTS = 200


# In[ ]:


def align(signals):
    window = np.ones(32) / 32
    for i in range(0, signals.shape[0], 3):
        signal = np.convolve(signals[i], window)[16:-15]  # this smoothing operation is slow with numba
        argmax = np.argmax(signal)
        signals[i:i+3] = np.concatenate((signals[i:i+3, argmax:], signals[i:i+3, :argmax]), axis=1)


# In[ ]:


@jit(nopython=True)
def get_feats(signal, n, splits):
    part_n = N_PARTS
    part_length = LENGTH // part_n
    new_sig = np.empty((n//splits, part_n, 3))  
    for i in range(part_n):
        segment = np.ascontiguousarray(  # numba asks for this
            signal[:, i*part_length:(i+1)*part_length]).astype(np.int16)
        mean_max_ent = np.empty((n//splits, 3))  # three features
        for j in range(n//splits):
            mean_max_ent[j, 0] = segment[j].mean()
            segment_diff = np.absolute(np.diff(segment[j]))
            mean_max_ent[j, 1] = np.log1p(segment_diff.max())  # log for outliers
            histogram = np.histogram(
                segment[j], bins=15, range=(-128., 127.))[0].astype(np.float64)
            histogram = histogram / histogram.sum()
            entropy = 0
            for k in range(15):
                if histogram[k] > 0:
                    entropy = entropy + histogram[k] * np.log(histogram[k])
            mean_max_ent[j, 2] = np.log1p(-entropy)  # log1p for outliers
        max_mean = np.max(mean_max_ent[:, 0])
        min_mean = np.absolute(np.min(mean_max_ent[:, 0]))
        mean_max_ent[:, 0] = mean_max_ent[:, 0] / (2*np.max(np.array([max_mean, min_mean])))
        new_sig[:, i] = mean_max_ent
    return new_sig


# In[ ]:


def save_data(train_test):
    if not train_test:
        address = ADDRESS_TRAIN
        n = N_TRAIN
        splits = SPLITS_TRAIN
        test_adjust = 0
    else:
        address = ADDRESS_TEST
        n = N_TEST
        splits = SPLITS_TEST
        test_adjust = N_TRAIN
        
    def get_signals():
        new_sigs = np.empty((n, N_PARTS, 3))
        for i in tqdm(range(splits)):
            start_index = i*(n//splits)
            end_index = (i+1)*(n//splits)
            cols = [str(k+test_adjust) for k in range(start_index, end_index)]
            signals_split = pd.read_parquet(address, columns=cols).values.T    
#             align(signals_split)  # this starts signals in groups of three phases at the same point
            new_sigs[start_index:end_index] = get_feats(signals_split, n, splits)
            del signals_split
            gc.collect()
        for i in range(1, 3):
            if np.max(new_sigs[:, :, i]) != 0:
                new_sigs[:, :, i] = new_sigs[:, :, i] / np.max(new_sigs[:, :, i])
                plt.hist(new_sigs[:, :, i].flatten())
                plt.show()
        plt.hist(new_sigs[:, :, 0].flatten())
        plt.show()
        return new_sigs

    comb_feats = get_signals()
    np.save('aligned_comb_feats_{}.npy'.format(str(train_test)), comb_feats)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'save_data(1)  # test set\nsave_data(0)  # train set')


# In[ ]:


signals = np.load('aligned_comb_feats_0.npy')
print(signals.shape)
signals = signals.reshape((N_PARTS*N_TRAIN//P, 9))
print(signals.shape)
signals = pd.DataFrame(signals)
signals.describe()  # three features for three phases is nine features per signal


# In[ ]:




