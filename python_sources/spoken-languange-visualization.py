#!/usr/bin/env python
# coding: utf-8

# Notebook ini dibangun menggunakan kaggle kernel, karena data yang sangat besar (15GB) dan resource belum mendukung juga kaggle kernel tidak cukup menangani semua data
# 
# Dalam notebook ini hanya memvisualisasi 1000 data pertama, belum sampai pada tahap modelling.

# In[ ]:


get_ipython().system('pip install soundfile')


# In[ ]:


import os
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt


# In[ ]:


train_path = '../input/train/train/'
test_path = '../input/test/test/'


# In[ ]:


filename = 'de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.flac'


# In[ ]:


data, samplerate = sf.read(train_path+filename)


# In[ ]:


data


# In[ ]:


freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')
plt.pcolormesh(time, freq, Sxx)

# add axis labels
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


# In[ ]:


Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)

# add axis labels
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


# In[ ]:


data


# In[ ]:


plt.plot(data)


# Karena resource pada kaggle kernel tidak cukup, maka hanya digunakan 1000 batch data

# In[ ]:


label = []
for filename in os.listdir(train_path)[:1000]:
    label.append(filename[:2])


# In[ ]:


gender = []
for filename in os.listdir(train_path)[:1000]:
    gender.append('male' if filename[3:4]=='m' else 'female')


# In[ ]:


sound_type = []
for filename in os.listdir(train_path)[:1000]:
    sound_type.append('noise' if 'noise' in filename else 'pitch' if 'pitch' in filename else 'speed' if 'speed' in filename else 'notype')


# In[ ]:


file = []
for filename in os.listdir(train_path)[:1000]:
    file.append(filename)


# In[ ]:


series = []
length = []
for filename in os.listdir(train_path)[:1000]:
    flac, samplerate = sf.read(train_path+filename)
    series.append(flac)
    length.append(samplerate)


# In[ ]:


data = {'Gender':gender,
        'filename':file,
           'Sound_type': sound_type,
       'languange':label,
       'series': series,
       'length': length}


# In[ ]:


df = pd.DataFrame(data)


# In[ ]:


len(df)


# In[ ]:


df = pd.DataFrame(data)


# In[ ]:


df.head()


# In[ ]:


df_male_en = df[(df['Gender']=='male') & (df['languange']=='en')].reset_index(drop=True)
df_male_es = df[(df['Gender']=='male') & (df['languange']=='es')].reset_index(drop=True)
df_male_de = df[(df['Gender']=='male') & (df['languange']=='de')].reset_index(drop=True)
df_female_en = df[(df['Gender']=='female') & (df['languange']=='en')].reset_index(drop=True)
df_female_es = df[(df['Gender']=='female') & (df['languange']=='es')].reset_index(drop=True)
df_female_de = df[(df['Gender']=='female') & (df['languange']=='de')].reset_index(drop=True)


# The first 40 male with english language waveform

# In[ ]:


fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in range(40):
    ax[i//4, i%4].plot(df_male_en['series'][i])
    ax[i//4, i%4].set_title(df_male_en['filename'][i][:-4], fontsize=5)
    ax[i//4, i%4].get_xaxis().set_ticks([])


# The first 40 male with espanol language waveform

# In[ ]:


fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in range(40):
    ax[i//4, i%4].plot(df_male_es['series'][i])
    ax[i//4, i%4].set_title(df_male_es['filename'][i][:-4], fontsize=5)
    ax[i//4, i%4].get_xaxis().set_ticks([])


# The first 40 male with deutch language waveform

# In[ ]:


fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in range(40):
    ax[i//4, i%4].plot(df_male_de['series'][i])
    ax[i//4, i%4].set_title(df_male_de['filename'][i][:-4], fontsize=5)
    ax[i//4, i%4].get_xaxis().set_ticks([])


# The first 40 female with english language waveform

# In[ ]:


fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in range(40):
    ax[i//4, i%4].plot(df_female_en['series'][i])
    ax[i//4, i%4].set_title(df_female_en['filename'][i][:-4], fontsize=5)
    ax[i//4, i%4].get_xaxis().set_ticks([])


# The first 40 female with espanol language waveform

# In[ ]:


fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in range(40):
    ax[i//4, i%4].plot(df_female_es['series'][i])
    ax[i//4, i%4].set_title(df_female_es['filename'][i][:-4], fontsize=5)
    ax[i//4, i%4].get_xaxis().set_ticks([])


# The first 40 female with deutch language waveform

# In[ ]:


fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in range(40):
    ax[i//4, i%4].plot(df_female_de['series'][i])
    ax[i//4, i%4].set_title(df_female_de['filename'][i][:-4], fontsize=5)
    ax[i//4, i%4].get_xaxis().set_ticks([])


# In[ ]:




