#!/usr/bin/env python
# coding: utf-8

# ***
# <br>
# <font size=7 color=darkgray><b>
# Freesound Audio Tagging 2019
# </b></font>
# <br>

# <img src="https://annotator.freesound.org/static/img/freesound_logo_color.png" alt="Drawing" style="width: 500px;"/>

# <font color=crimson size=4><b> More to Go. Stay tuned. </b></font>
# <font color=crimson size=2><b> Apr. 15. 2019 updated </b></font>
# ---
# In previous kernel I have posted, I did not consider multi labels for simplicity.  
# [Beginner's Guide to Auidio Data 2](https://www.kaggle.com/maxwell110/beginner-s-guide-to-audio-data-2)  
# Here I will do simple explorations,  
# 
# - wav sounds
# - spectrogram
# - co-occurence  
#   
# in multi labeled records.
# 
# # Contents
# 1. [LOAD PACKAGES](#lp)
# 2. [LOAD DATA](#ld)
# 3. [Single Labeled](#singlel)  
# 4. [Multi Labeled](#multil)  
#     4.1 [With 2 labels](#2l)  
#     4.2 [With 3 labels](#3l)  
#     4.3 [With 4 labels](#4l)  
#     4.4 [With 5 labels](#5l)  
#     4.5 [With 6 labels](#6l)  
#     4.6 [With 7 labels](#7l)  
#     4.7 [Co-Occurence](#coo)  
# ***
# 

# <a id="lp"></a>
# # <center>1. LOAD PACKAGES</center>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=1.2)

import warnings

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

plt.style.use('ggplot')

import missingno as msno
import datetime as dt
import IPython.display as ipd
import librosa

from random import sample
from collections import OrderedDict
from datetime import timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(2019)


# In[ ]:


SAMPLE_RATE = 44100


# <a id="ld"></a>
# # <center>2. LOAD DATA</center>

# In[ ]:


train = pd.read_csv("../input/train_curated.csv")
train['is_curated'] = True
train_noisy = pd.read_csv('../input/train_noisy.csv')
train_noisy['is_curated'] = False
train = pd.concat([train, train_noisy], axis=0)
del train_noisy


# In[ ]:


train.sample(5)


# In[ ]:


train['n_label'] = train.labels.str.split(',').apply(lambda x: len(x))


# In[ ]:


train.query('is_curated == True').n_label.value_counts()


# * Some records in curated train data have 2, 3, 4, 6 labels.  
# * 5752 labels in curated train data

# In[ ]:


train.query('is_curated == False').n_label.value_counts()


# * Some records in noisy train data have 2, 3, 4, 5, 6, 7 labels.
# * 24000 labels

# <a id="singlel"></a>
# # <center> 3. Single Labeled </center>

# In[ ]:


cat_gp = train[train.n_label == 1].groupby(
    ['labels', 'is_curated']).agg({
    'fname':'count'
}).reset_index()
cat_gpp = cat_gp.pivot(index='labels', columns='is_curated', values='fname').reset_index().set_index('labels')

plot = cat_gpp.plot(
    kind='barh',
    title="Number of samples per category",
    stacked=True,
    color=['deeppink', 'darkslateblue'],
    figsize=(15,20))
plot.set_xlabel("Number of Samples", fontsize=20)
plot.set_ylabel("Label", fontsize=20);


# Let's check curated and noisy data with same label.

# In[ ]:


# sampling an audio in train_curated
samp = train[(train.n_label == 1) & (train.is_curated == True)].sample(1)
print(samp.labels.values[0])
ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))


# In[ ]:


# sampling an audio in train_noisy
samp_n = train[(train.n_label == 1) & 
               (train.is_curated == False) & 
               (train.labels == samp.labels.values[0])].sample(1)
print(samp_n.labels.values[0])
ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0]))


# Could you confirm the same audio in both curated and noisy ? If so, maybe you are lucky :)

# In[ ]:


# trim silent part
wav, _ = librosa.core.load(
    '../input/train_curated/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
wav_n, _ = librosa.core.load(
    '../input/train_noisy/{}'.format(samp_n.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr_n = librosa.effects.trim(wav_n)[0]
print('After trimmed curated wav: {:,}/{:,}'.format(len(wav_tr), len(wav)))
print('After trimmed noisy wav: {:,}/{:,}'.format(len(wav_tr_n), len(wav_n)))


# In[ ]:


melspec = librosa.feature.melspectrogram(
    librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2),
    sr=SAMPLE_RATE/2,
    n_fft=1764,
    hop_length=220,
    n_mels=64
)
logmel = librosa.core.power_to_db(melspec)

melspec_n = librosa.feature.melspectrogram(
    librosa.resample(wav_tr_n, SAMPLE_RATE, SAMPLE_RATE/2),
    sr=SAMPLE_RATE/2,
    n_fft=1764,
    hop_length=220,
    n_mels=64
)
logmel_n = librosa.core.power_to_db(melspec_n)


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(15, 10))
for i, l in enumerate([logmel, logmel_n]):
    if i==0: 
        ax[i].set_title('curated {}'.format(samp.labels.values[0]))
    else:
        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))
    im = ax[i].imshow(l, cmap='Spectral', interpolation='nearest',
                      aspect=l.shape[1]/l.shape[0]/5)


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, 
                            sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
mfcc_n = librosa.feature.mfcc(wav_tr_n, 
                              sr=SAMPLE_RATE, 
                              n_fft=1764,
                              hop_length=220,
                              n_mfcc=64)

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
for i, m in enumerate([mfcc, mfcc_n]):
    if i==0: 
        ax[i].set_title('curated {}'.format(samp.labels.values[0]))
    else:
        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))
    im = ax[i].imshow(m, cmap='Spectral', interpolation='nearest',
                      aspect=m.shape[1]/m.shape[0]/5)


# <a id="multil"></a>
# # <center> 4. Multi Labeled </center>

# There are a lot of kinds of muti labels.  

# In[ ]:


print('Unique number of multi label : {}'.format(train.loc[train.n_label > 1, 'labels'].nunique()))
print('Unique number of multi label in curated data : {}'.format(
    train.loc[(train.n_label > 1) & (train.is_curated == True), 'labels'].nunique()))
print('Unique number of multi label in noisy data : {}'.format(
    train.loc[(train.n_label > 1) & (train.is_curated == False), 'labels'].nunique()))    


# Let's visualize only multi labeled data in curated train.

# In[ ]:


cat_gp = train[(train.n_label > 1) & (train.is_curated == True)].groupby(
    'labels').agg({'fname':'count'})
cat_gp.columns = ['counts']

plot = cat_gp.sort_values(ascending=True, by='counts').plot(
    kind='barh',
    title="Number of Audio Samples per Category",
    color='deeppink',
    figsize=(15,30))
plot.set_xlabel("Number of Samples", fontsize=20)
plot.set_ylabel("Label", fontsize=20);


# 1.   Half of multi-labeled categories have only `1` record in curated train data.  
# 2.   Similar audios tend to appear at once (e.g. Acoustic guitar, Strum).  
#   
# Let's listen multi-labeled audios and check log-mel and MFCC.  

# <a id="2l"></a>
# ## 4.1 With 2 labels

# In[ ]:


label_set = set(train.loc[(train.n_label == 2) & (train.is_curated == True), 'labels']) & set(
    train.loc[(train.n_label == 2) & (train.is_curated == False), 'labels'])

label_samp = np.random.choice(list(label_set), 1)[0]
samp = train[(train.labels == label_samp) & (train.is_curated == True)].sample(1)
print(label_samp)
ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))


# In[ ]:


# sampling an audio in train_noisy
samp_n = train[(train.labels == label_samp) & (train.is_curated == False)].sample(1)
print(samp_n.labels.values[0])
ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0]))


# In[ ]:


# trim silent part
wav, _ = librosa.core.load(
    '../input/train_curated/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
print('After trimmed curated wav: {:,}/{:,}'.format(len(wav_tr), len(wav)))

wav_n, _ = librosa.core.load(
    '../input/train_noisy/{}'.format(samp_n.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr_n = librosa.effects.trim(wav_n)[0]
print('After trimmed noisy wav: {:,}/{:,}'.format(len(wav_tr_n), len(wav_n)))


# In[ ]:


melspec = librosa.feature.melspectrogram(
    librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2),
    sr=SAMPLE_RATE/2,
    n_fft=1764,
    hop_length=220,
    n_mels=64
)
logmel = librosa.core.power_to_db(melspec)

melspec_n = librosa.feature.melspectrogram(
    librosa.resample(wav_tr_n, SAMPLE_RATE, SAMPLE_RATE/2),
    sr=SAMPLE_RATE/2,
    n_fft=1764,
    hop_length=220,
    n_mels=64
)
logmel_n = librosa.core.power_to_db(melspec_n)


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(15, 10))
if samp_n.labels.values[0] == samp.labels.values[0]:
    for i, l in enumerate([logmel, logmel_n]):
        if i==0: 
            ax[i].set_title('curated {}'.format(samp.labels.values[0]))
        else:
            ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))
        im = ax[i].imshow(l, cmap='Spectral', interpolation='nearest',
                          aspect=l.shape[1]/l.shape[0]/5)


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, 
                            sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
mfcc_n = librosa.feature.mfcc(wav_tr_n, 
                              sr=SAMPLE_RATE, 
                              n_fft=1764,
                              hop_length=220,
                              n_mfcc=64)

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
for i, m in enumerate([mfcc, mfcc_n]):
    if i==0: 
        ax[i].set_title('curated {}'.format(samp.labels.values[0]))
    else:
        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))
    im = ax[i].imshow(m, cmap='Spectral', interpolation='nearest',
                      aspect=m.shape[1]/m.shape[0]/5)


# <a id="3l"></a>
# ## 4.2 With 3 labels

# In[ ]:


label_set = set(train.loc[(train.n_label == 3) & (train.is_curated == True), 'labels']) & set(
    train.loc[(train.n_label == 3) & (train.is_curated == False), 'labels'])

label_samp = np.random.choice(list(label_set), 1)[0]
samp = train[(train.labels == label_samp) & (train.is_curated == True)].sample(1)
print(label_samp)
ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))


# In[ ]:


# sampling an audio in train_noisy
samp_n = train[(train.labels == label_samp) & (train.is_curated == False)].sample(1)
print(samp_n.labels.values[0])
ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0]))


# I think the audios with multi label are more noisy (mean wrong labeled).

# In[ ]:


# trim silent part
wav, _ = librosa.core.load(
    '../input/train_curated/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
print('After trimmed curated wav: {:,}/{:,}'.format(len(wav_tr), len(wav)))

wav_n, _ = librosa.core.load(
    '../input/train_noisy/{}'.format(samp_n.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr_n = librosa.effects.trim(wav_n)[0]
print('After trimmed noisy wav: {:,}/{:,}'.format(len(wav_tr_n), len(wav_n)))


# In[ ]:


melspec = librosa.feature.melspectrogram(
    librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2),
    sr=SAMPLE_RATE/2,
    n_fft=1764,
    hop_length=220,
    n_mels=64
)
logmel = librosa.core.power_to_db(melspec)

melspec_n = librosa.feature.melspectrogram(
    librosa.resample(wav_tr_n, SAMPLE_RATE, SAMPLE_RATE/2),
    sr=SAMPLE_RATE/2,
    n_fft=1764,
    hop_length=220,
    n_mels=64
)
logmel_n = librosa.core.power_to_db(melspec_n)


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(15, 10))
if samp_n.labels.values[0] == samp.labels.values[0]:
    for i, l in enumerate([logmel, logmel_n]):
        if i==0: 
            ax[i].set_title('curated {}'.format(samp.labels.values[0]))
        else:
            ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))
        im = ax[i].imshow(l, cmap='Spectral', interpolation='nearest',
                          aspect=l.shape[1]/l.shape[0]/5)


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, 
                            sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
mfcc_n = librosa.feature.mfcc(wav_tr_n, 
                              sr=SAMPLE_RATE, 
                              n_fft=1764,
                              hop_length=220,
                              n_mfcc=64)

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
for i, m in enumerate([mfcc, mfcc_n]):
    if i==0: 
        ax[i].set_title('curated {}'.format(samp.labels.values[0]))
    else:
        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))
    im = ax[i].imshow(m, cmap='Spectral', interpolation='nearest',
                      aspect=m.shape[1]/m.shape[0]/5)


# <a id="4l"></a>
# ## 4.3 With 4 labels
# <b>
#     There are no common record with same multi labels in curated and noisy.  
#     So, here we will check only curated audios.
# </b>

# In[ ]:


samp = train[(train.n_label == 4) & (train.is_curated == True)].sample(1)
print(samp.labels.values[0])
ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))


# In[ ]:


wav, _ = librosa.core.load(
    '../input/train_curated/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))


# In[ ]:


wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)
melspec = librosa.feature.melspectrogram(wav_tr,
                                         sr=SAMPLE_RATE/2,
                                         n_fft=1764,
                                         hop_length=220,
                                         n_mels=64)
logmel = librosa.core.power_to_db(melspec)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',
          aspect=logmel.shape[1]/logmel.shape[0]/5)
ax.set_title('log-mel');


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',
          aspect=mfcc.shape[1]/mfcc.shape[0]/5)
ax.set_title('MFCC');


# <a id="5l"></a>
# ## 4.4 With 5 labels
# <b>
#     There are no record with 5 labels in curated data.
# </b>

# In[ ]:


samp = train[train.n_label == 5].sample(1)
print(samp.labels.values[0])
ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0]))


# The label seems valid ?

# In[ ]:


wav, _ = librosa.core.load(
    '../input/train_noisy/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))


# In[ ]:


wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)
melspec = librosa.feature.melspectrogram(wav_tr,
                                         sr=SAMPLE_RATE/2,
                                         n_fft=1764,
                                         hop_length=220,
                                         n_mels=64)
logmel = librosa.core.power_to_db(melspec)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',
          aspect=logmel.shape[1]/logmel.shape[0]/5)
ax.set_title('log-mel');


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',
          aspect=mfcc.shape[1]/mfcc.shape[0]/5)
ax.set_title('MFCC');


# <a id="6l"></a>
# ## 4.5 With 6 labels
# <b>
#     There are no common record with same multi labels in curated and noisy.  
#     So, here we will check only curated audios.
# </b>

# In[ ]:


samp = train[(train.n_label == 6) & (train.is_curated == True)].sample(1)
print(samp.labels.values[0])
ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))


# In[ ]:


wav, _ = librosa.core.load(
    '../input/train_curated/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))


# In[ ]:


wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)
melspec = librosa.feature.melspectrogram(wav_tr,
                                         sr=SAMPLE_RATE/2,
                                         n_fft=1764,
                                         hop_length=220,
                                         n_mels=64)
logmel = librosa.core.power_to_db(melspec)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',
          aspect=logmel.shape[1]/logmel.shape[0]/5)
ax.set_title('log-mel');


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',
          aspect=mfcc.shape[1]/mfcc.shape[0]/5)
ax.set_title('MFCC');


# <a id="7l"></a>
# ## 4.6 With 7 labels
# <b>
#     There are no record with 7 labels in curated.  
# </b>

# In[ ]:


samp = train[train.n_label == 7].sample(1)
print(samp.labels.values[0])
ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0]))


# In[ ]:


wav, _ = librosa.core.load(
    '../input/train_noisy/{}'.format(samp.fname.values[0]),
    sr=SAMPLE_RATE)
wav_tr = librosa.effects.trim(wav)[0]
print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))


# In[ ]:


wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)
melspec = librosa.feature.melspectrogram(wav_tr,
                                         sr=SAMPLE_RATE/2,
                                         n_fft=1764,
                                         hop_length=220,
                                         n_mels=64)
logmel = librosa.core.power_to_db(melspec)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',
          aspect=logmel.shape[1]/logmel.shape[0]/5)
ax.set_title('log-mel');


# In[ ]:


mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 
                            n_fft=1764,
                            hop_length=220,
                            n_mfcc=64)
fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',
          aspect=mfcc.shape[1]/mfcc.shape[0]/5)
ax.set_title('MFCC');


# <b><font color=gray size=6> Findings </font></b>
#   
# 1. Multi-labeled spectrograms look to have some patterns at the same timestamp or in different time periods.  
# 2. Signals of spectrograms are mixiture of multiple sounds. I think this is the intrinsic difference from image data.  Band pass filters may help to separate multiple sounds and we can use those separated spectrogram data as an input to CNN models at once.
# 3. A lot of noisy data seem to have wrong labels. In the previous competition, using the robust loss function to suppress the effect of mislabeled data was one of the important points to get high score. We must care of that again.

# <a id="coo"></a>
# ## 4.7 Co-Occurrence in curated train data
# 
# We confirmed that noisy train data seems to have a lot of mislabeled records.  
# Then we will use only curated train data to compute co-occurrence.

# In[ ]:


test = pd.read_csv('../input/sample_submission.csv')
for c in test.columns[1:]:
    cc = c.replace('(', '\(').replace(')', '\)')
    train.loc[:, c] = train['labels'].str.contains(cc)
    if (train.loc[:, c] > 1).sum():
        raise Exception('label key "{}" are duplicated !'.format(c))


# In[ ]:


train.head()


# Before computing co-occurrence, let's check the number of each label in curated and noisy data.

# In[ ]:


tmp = train.loc[train.is_curated == True, 
                'Accelerating_and_revving_and_vroom':].sum(axis=0).to_frame().T
tmp = pd.concat(
    [tmp, train.loc[train.is_curated == False, 
                    'Accelerating_and_revving_and_vroom':].sum(axis=0).to_frame().T]
)
tmp['total_label'] = tmp.loc[:, 'Accelerating_and_revving_and_vroom':].sum(axis=1)
tmp.index = ['curated', 'noisy']
tmp


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 20))
tmp.iloc[:, :-1].T.plot.barh(color=['deeppink', 'darkslateblue'], ax=ax);


# After considering multi label, we can see all kinds of labels in curated and noisy data.

# In[ ]:


sns.set(style="white")
sns.set(font_scale=1)
train_cor = train.loc[train.is_curated == True, test.columns[1:]].corr()
mask = np.zeros_like(train_cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(train_cor, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .1});


# For easy-to-see visuzalization,  
# we select only multi-labeled records and use seaborns cluastermap.  

# In[ ]:


multi_label = test.columns[1:][train.loc[
    (train.n_label > 1) & (train.is_curated == True), test.columns[1:]].sum() > 0]


# In[ ]:


sns.set(font_scale=1.7)
train_cor = train.loc[
    (train.n_label > 1) & (train.is_curated == True), multi_label].corr()
mask = np.zeros_like(train_cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.clustermap(train_cor, metric='correlation',
               cmap=cmap, center=0, linewidths=.5, figsize=(30, 30));


# <center><b><font size=4 color=deepskyblue>
#     Thanks for reading.  
#     <br>
#     Happy Kaggling!
#     </font></b></center>
# <br>

# # <center> EOF </center>
