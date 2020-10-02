#!/usr/bin/env python
# coding: utf-8

# Hi, everyone. This is a simple kernel that samples sounds from both the curated and noisy datasets and displays them as audible audios and their corresponding mel-scaled power spectrograms.
# 
# I make this kernel because it might be useful to inspect data characteristics, for example, it is noticeable that files in the noisy dataset are often cacophony and contain not only labeled sounds but also other sounds as well.

# You can fork this kernel and change these following parameters to examine the data.

# In[ ]:


SAMPLES = 3 # the number of displayed samples
SEED = 2019 # the seed that is used to generate samples
LABEL = 'Bark' # the label to examine (all labels are listed in the next cell output)


# In[ ]:


import os
from pathlib import Path

import numpy as np
import pandas as pd

import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt


# In[ ]:


test_df = pd.read_csv('../input/sample_submission.csv')
labels = test_df.columns[1:].tolist()
print(labels)


# The functions about audios and spectrograms are from [this great kernel](https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data) by [daisukelab](https://www.kaggle.com/daisukelab).

# In[ ]:


class conf:
    sampling_rate = 44100
    duration = 2 # sec
    hop_length = 347 * duration # sampling_rate * duration must be reduced to 128 which is the image width of duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128 # so the height of the image is 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration
    
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)
    return y

def audio_to_melspectrogram(conf, audio, scale=1):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=int(conf.n_fft*scale),
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# In[ ]:


curated_df = pd.read_csv('../input/train_curated.csv')
noisy_df = pd.read_csv('../input/train_noisy.csv')

curated_df['label_list'] = curated_df['labels'].str.split(',')
noisy_df['label_list'] = noisy_df['labels'].str.split(',')


# In[ ]:


def show_samples(df, source_folder):
    samples = df[df['label_list'].apply(lambda x: LABEL in x)].sample(SAMPLES, random_state=SEED)
    for i, row in samples.iterrows():
        print(f'{i} {row.fname} {row.labels}')
        audio = read_audio(conf, source_folder/row.fname, trim_long_data=False)
        ipd.display(ipd.Audio(audio, rate=conf.sampling_rate))
        spectrogram = audio_to_melspectrogram(conf, audio)
        plt.figure(figsize=(15, 3))
        plt.imshow(spectrogram)
        plt.show()


# In[ ]:


show_samples(curated_df, Path('../input/train_curated'))


# In[ ]:


show_samples(noisy_df, Path('../input/train_noisy'))

