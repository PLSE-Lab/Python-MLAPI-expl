#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy import fftpack
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Intro
# The acoustic data can be analyzed in frequency space. It gives a pretty different feeling and the information that exists in the data.
# 
# I wanted to show you 4 things:
# 1. Frequency representation of the data
# 2. Some nice compresion
# 3. Different "compresion" - frequency bins
# 4. Data size can be reduced easily with no lost in the frequeny information
# 
# Let's take a look at what we've got here.

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=10000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("train shape", train.shape)
train.head()


# I set the important parameter here, that will be crucial in proper calculations

# In[ ]:


sampling_rate = 4000000


# ## Frequency representation of the data

# The pitch perception of the ear is proportional to the logarithm of frequency rather than to frequency itself. We can't hear most of the frequencies that exist here (0 - 2000000 vs 0-20000 Hz), but watching the frequencies in logarithmic scale is much nicer.
# 
# So I define _log_specgram_ funcion to calcuate the logarithm of STFT, and a funciton to visualize it.

# In[ ]:


def log_specgram(data, sample_rate, nperseg=2000, noverlap=1000, eps=1e-10, dct=False):
    freqs, times, spec = signal.spectrogram(data,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    spec = np.log(spec).astype(np.float32)
    if dct:
        spec = fftpack.dct(spec, type=2, axis=0, norm='ortho')
    return freqs, times, spec


def plot_specgram(data, sample_rate, final_idx, init_idx=0, step=1, nperseg=2000, 
                  noverlap=1000, dct=False, title='', subsampling=False):
    idx = [i for i in range(init_idx, final_idx, step)]
    acoustic_data = data.iloc[idx].acoustic_data.values
    if subsampling:
        acoustic_data = acoustic_data[::subsampling]
    freqs, times, spectrogram = log_specgram(acoustic_data, sample_rate, 
                                             nperseg=nperseg, noverlap=noverlap, dct=dct)

    plt.figure(figsize=(10, 8))
    plt.imshow(spectrogram, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.title(title)
    plt.ylabel('Freqs in Hz')
    plt.xlabel('Seconds')
    plt.show()


# I also borrowed a function from the kernel I really liked:
# https://www.kaggle.com/jsaguiar/seismic-data-exploration

# In[ ]:


def single_timeseries(final_idx, init_idx=0, step=1, title="",
                      color1='orange', color2='blue', subsampling=False):
    idx = [i for i in range(init_idx, final_idx, step)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('index')
    ax1.set_ylabel('Acoustic data')
    ax2.set_ylabel('Time to failure')
    acoustic_data = train.iloc[idx].acoustic_data.values
    time_to_failure = train.iloc[idx].time_to_failure.values
    if subsampling:
        acoustic_data = acoustic_data[::subsampling]
        time_to_failure = time_to_failure[::subsampling]

    p1 = sns.lineplot(data=acoustic_data, ax=ax1, color=color1)
    p2 = sns.lineplot(data=time_to_failure, ax=ax2, color=color2)


# Let's plot some data:

# In[ ]:


plot_specgram(train, sampling_rate, 100000, title='Specgram of first hundred thousand rows')


# In[ ]:


single_timeseries(100000, title="First hundred thousand rows")


# Earthquake data:

# In[ ]:


plot_specgram(train, sampling_rate, final_idx=6000000, init_idx=5000000, 
              title='Specgram of earthquake')


# In[ ]:


single_timeseries(final_idx=6000000, init_idx=5000000, title="Five to six million index")


# ## DCT - Discrete Cosinous Transform
# 
# One of the steps in the popular algorithm in acoustic signal analysis, MFCC (Mel-Frequency-Cepstral-Coefficients), is Discrete Cosine Transform. According to [wikipedia](https://en.wikipedia.org/wiki/Discrete_cosine_transform), it can decorrelate some signals very well. It happens fast and online. You can think about like doing some PCA on data.

# In[ ]:


plot_specgram(train, sampling_rate, 10000, title='Specgram of first hundred thousand rows', dct=True)


# It's up to you how and if to use it. It works very well in speech recognition. You can plot fewer components to choose the threshold for the number of it.

# ## Data size reduction - subsampling

# You can see that there's a lot of 'blank' space in the spectrogram. It looks like most of the high frequencies are just noise. We can subsample the data by, let's say 4, to leave only the meaningful part of the signal. Without going into details, we just need to take every 4th sample and change the sampling rate to 1M.
#  I'm gonna plot some part of the data, but subsampled. You can compare it to the original format.
# 

# In[ ]:


single_timeseries(80000, title="First eighty thousand rows")


# In[ ]:


plot_specgram(train, sampling_rate, 80000, title='Specgram of first eighty thousand rows')


# In[ ]:


single_timeseries(80000, title="First eighty thousand rows - subsampled", subsampling=4)


# In[ ]:


plot_specgram(train, sample_rate=1000000, final_idx=80000, nperseg=500, noverlap=250, 
              title='Subsampled specgram of first eighty thousand rows', subsampling=4)


# Skipping every 3 samples, we can carry the same frequency information, having the data size 4 times smaller!

# I hope my presentation shows you some interesting ideas. Don't forget to upvote if you like it.
