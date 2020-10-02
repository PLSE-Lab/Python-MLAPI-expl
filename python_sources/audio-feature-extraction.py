#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from scipy.io import wavfile as wav
print(os.listdir("../input"))
import matplotlib.pyplot as plt
from IPython.display import Audio
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import signal
import numpy as np
import librosa
from keras.preprocessing.sequence import pad_sequences
import sklearn.metrics
import glob
import json
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[2]:


df_c = pd.read_csv('../input/train_curated.csv')
df_c.info()


# In[3]:


df_c['labels'].describe()


# In[4]:


# feature extraction 
def extract_feature(path):
    X, sample_rate = librosa.load(path)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(df, loc='../input/train_curated/'):
    # n: number of classes
    features = np.empty((0,193))
    for idx, row in tqdm(df.iterrows()):
        f = loc + row['fname']
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
    return np.array(features)


# In[7]:


d = parse_audio_files(df_c)


# In[10]:


import pickle
try:
    with open('data.pkl', 'wb') as file:
        pickle.dump(d,file)
except Exception as e:
    print(e)


# In[11]:


d.dump('d.pkl')


# In[ ]:


#one hot encoding
y = []
for index, row in df_c.iterrows():
    labels = row['labels'].split(',')
    y.append(labels)

# Create MultiLabelBinarizer object
one_hot = MultiLabelBinarizer()

# One-hot encode label data
y = one_hot.fit_transform(y)

