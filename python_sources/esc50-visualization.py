#!/usr/bin/env python
# coding: utf-8

# # Retriving and visualizing the data
# 
# In this notebook we (1) demonstrate how to access the raw 'wav' files with python and (2) show a simple visualization of some audio files.
# 
# 
# ## Retriving the Data
# To access the data, we use the utils.py present in the dataset.
# 
# **ESC50** is a Class usefull to access the data as 16000 or 44100KHz raw waves.  
# You call ESC50 with the desired data augmentation desired:  
# `ESC50(only_ESC10, folds, randomize, audio_rate, strongAugment, 
#       pad, inputLength, random_crop, mix, normalize)`   
# It has 2 generator included :
# - `_data_gen` to retrieve audio samples, one at a time. 
# - `batch_gen(batch_size)` to retrieve batches of audio samples (calls the former).

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../input/'))
from utils import ESC50

train_splits = [1,2,3,4]
test_split = 5

shared_params = {'csv_path': '../input/esc50.csv',
                 'wav_dir': '../input/audio/audio',
                 'dest_dir': '../input/audio/audio/16000',
                 'audio_rate': 16000,
                 'only_ESC10': True,
                 'pad': 0,
                 'normalize': True}

train_gen = ESC50(folds=train_splits,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=True,
                  **shared_params).batch_gen(16)

test_gen = ESC50(folds=[test_split],
                 randomize=False,
                 strongAugment=False,
                 random_crop=False,
                 inputLength=4,
                 mix=False,
                 **shared_params).batch_gen(16)

X, Y = next(train_gen)
X.shape, Y.shape


# ## Get the classes
# We now compute the spectrogram of each audio sample we have collected before (in data) and plot it next to its raw wave.

# In[ ]:


df = pd.DataFrame.from_csv('../input/esc50.csv')
classes = df[['target', 'category']].as_matrix().tolist()
classes = set(['{} {}'.format(c[0], c[1]) for c in classes])
classes = np.array([c.split(' ') for c in classes])
classes = {k: v for k, v in classes}


# ## Visualizing the Data
# We now compute the spectrogram of each audio sample we have collected before (in data) and plot it next to its raw wave.

# In[ ]:


import scipy
from scipy import signal
import IPython.display as ipd

fig, axs = plt.subplots(2, 5, figsize=(13, 4))
for idx in range(5):
    i, j = int(idx / 5), int(idx % 5)
    x = X[idx]
    sampleFreqs, segmentTimes, sxx = signal.spectrogram(x[:, 0], 16000)
    axs[i*2][j].pcolormesh((len(segmentTimes) * segmentTimes / segmentTimes[-1]),
                         sampleFreqs,
                         10 * np.log10(sxx + 1e-15))
    #axs[i*2][j].set_title(classes[seen_classes[idx]])
    axs[i*2][j].set_axis_off()
    axs[i*2+1][j].plot(x)
    #axs[i*2+1][j].set_axis_off()
    
plt.show()

for idx in range(5):
    x = X[idx]
    ipd.display(ipd.Audio(x[:, 0], rate=16000))


# In[ ]:




