#!/usr/bin/env python
# coding: utf-8

# # Mel-Frequency Ceptral Coeffienents(MFCC) feature extraction for Sound Classification
# 
# Using the MFCC feature for sound classification like the [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition/overview) is common. It takes few hours for Cornell Birdcall Identification datasets. I will share extracted feature as dataset after the execution in colab. In this notebook, I just use 3 mp3 files for each bird class. (check the LIMIT variable)
# 
# Please enjoy it and don't forget to vote it. Feel free to give an advice.

# ## Mel-Frequency Cepstral Coefficients (MFCCs)
# 
# ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FTsu71%2FbtqETBgoxsP%2F7rgu73Uyc3isPddR9q1ZOK%2Fimg.png)
# 
# The log-spectrum already takes into account perceptual sensitivity on the magnitude axis, by expressing magnitudes on the logarithmic-axis. The other dimension is then the frequency axis. 
# 
# There exists a multitude of different criteria with which to quantify accuracy on the frequency scale and there are, correspondingly, a multitude of perceptually motivated frequency scales including the equivalent rectangular bandwidth (ERB) scale, the Bark scale, and the mel-scale. Probably through an abritrary choice mainly due to tradition, in this context we will focus on the mel-scale. This scale describes the perceptual distance between pitches of different frequencies. 
# 
# Though the argumentation for the MFCCs is not without problems, it has become the most used feature in speech and audio recognition applications. It is used because it works and because it has relatively low complexity and it is straightforward to implement. Simply stated,
# 
# if you're unsure which inputs to give to a speech and audio recognition engine, try first the MFCCs.
# 
# ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FlR19O%2FbtqETBgoAUx%2F8mcBOUb3mJkHW92sGyGMB0%2Fimg.png)
# 
# The beneficial properties of the MFCCs include:
# 
# Quantifies the gross-shape of the spectrum (the spectral envelope), which is important in, for example, identification of vowels. At the same time, it removes fine spectral structure (micro-level structure), which is often less important. It thus focuses on that part of the signal which is typically most informative.
# Straightforward and computationally reasonably efficient calculation.
# Their performance is well-tested and -understood.
# Some of the issues with the MFCC include:
# 
# The choice of perceptual scale is not well-motivated. Scales such as the ERB or gamma-tone filterbanks might be better suited. However, these alternative filterbanks have not demonstrated consistent benefit, whereby the mel-scale has persisted.
# MFCCs are not robust to noise. That is, the performance of MFCCs in presence of additive noise, in comparison to other features, has not always been good. 
# The choice of triangular weighting filters wk,h is arbitrary and not based on well-grounded motivations. Alternatives have been presented, but they have not gained popularity, probably due to minor effect on outcome.
# The MFCCs work well in analysis but for synthesis, they are problematic. Namely, it is difficult to find an inverse transform (from MFCCs to power spectra) which is simultaneously unbiased (=accurate) and congruent with its physical representation (=power spectrum must be positive).
# 
# ref: https://wiki.aalto.fi/display/ITSP/Cepstrum+and+MFCC <br/>
# ref: https://melon1024.github.io/ssc/

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import glob
import librosa
import librosa.display
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.utils import np_utils

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


LIMIT = 3


# In[ ]:


get_ipython().system('ls ../input/birdsong-recognition')


# In[ ]:


df_train = pd.read_csv('../input/birdsong-recognition/train.csv')
df_train


# In[ ]:


get_ipython().system('ls ../input/birdsong-recognition/train_audio')

train_dir = '../input/birdsong-recognition/train_audio'
test_idr = '../input/birdsong-recognition/test_audio'


# # Extract Feature using MFCC()

# In[ ]:


def mfcc_extract(filename):
    try:
        y, sr  = librosa.load(filename, sr = 44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(0.02*sr),hop_length=int(0.01*sr))
        return mfcc
    except:
        return


# In[ ]:


def parse_audio_files(parent_dir, sub_dirs, limit):
    labels = []
    features = []
    for label, sub_dir in enumerate(tqdm(sub_dirs)):
        i = 0
        for fn in glob.glob(os.path.join(parent_dir,sub_dir,"*.mp3")):
            if i >= limit:
                break
            features.append(mfcc_extract(fn))
            labels.append(label)
            i+=1
    return features, labels


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_cat_dirs = glob.glob(train_dir+'/*')\ntrain_cat = []\nfor cat_dir in train_cat_dirs:\n    tmp = cat_dir.split('/')[-1]\n    train_cat.append(tmp)\nprint('the number of kinds:', len(train_cat))\n\nclass_num = len(train_cat)\nfeatures, labels = parse_audio_files(train_dir, train_cat, LIMIT)")


# In[ ]:


print(len(features))
print(features[0].shape)


# In[ ]:


# plot few features

fig = plt.figure(figsize=(28,24))
for i,mfcc in enumerate(tqdm(features[:100])):
    if i%40 < 3 : 
        sub = plt.subplot(10,3,i%40+3*(i/40)+1)
        librosa.display.specshow(mfcc,vmin=-700,vmax=300)
        if ((i%40+3*(i/40)+1)%3==0) : 
            plt.colorbar()
        sub.set_title(train_cat[labels[i]])
plt.show()  


# In[ ]:


df_submission = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
df_submission.to_csv('submission.csv', index = None)

