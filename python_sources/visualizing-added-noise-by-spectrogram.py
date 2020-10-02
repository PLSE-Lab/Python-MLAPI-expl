#!/usr/bin/env python
# coding: utf-8

# As it has been pointed out (like in [this amazing kernel](https://www.kaggle.com/friedchips/on-markov-chains-and-the-competition-data) by [Markus F](https://www.kaggle.com/friedchips), the signals in this competition data are simple Markov processes combined with Gaussian noise. 
# 
# Here I tried to visualize what's going on in the signal in the Fourier space by computing the Fourier transform and plot the spectrogram. The aim was to clearly visualize the nature of the added noise and see if there is any correlation between Fourier features of the signal and the open channels. If the noise is a pure white noise where the power is equal across different frequencies, we don't have a high hope to use some Fourier features to make a good prediction. If there is a systematic pattern in the noise, well, let's think about what we can do:) 

# In[ ]:


import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fftshift
from tqdm import tqdm_notebook as tqdm

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('fivethirtyeight')


# Let's load the data and split it into the 10 separate measurement sequences. I use cleaned data from [this kernel](https://www.kaggle.com/friedchips/clean-removal-of-data-drift).

# In[ ]:


# load data
df_train = pd.read_csv("../input/data-without-drift/train_clean.csv")
train_time   = df_train["time"].values.reshape(-1,500000)
train_signal = df_train["signal"].values.reshape(-1,500000)
train_opench = df_train["open_channels"].values.reshape(-1,500000)
# df_test = pd.read_csv("../input/data-without-drift/test_clean.csv")
# test_time   = df_test["time"].values.reshape(-1,500000)
# test_signal = df_test["signal"].values.reshape(-1,500000)


# In[ ]:


# # sample data for quick test
# train_time = train_time[:, ::100]
# train_signal = train_signal[:, ::100]
# train_opench = train_opench[:, ::100]


# In[ ]:


train_signal.shape


# I plot (1) time vs open channels, (2) spectrogram, (3) power spectrum to hopefully get insight into the signal and noise.

# In[ ]:


def spectrogram_plot(train_signal, train_opench, i):
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), 
                           gridspec_kw={"height_ratios": [1, 3, 1]})
    ax = ax.flatten()
    
    # open channels
    ax[0].plot(np.arange(500_000), train_opench[i], lw=0.05, color='r')
    ax[0].set_title(f"batch {i}")
    ax[0].set_ylabel("open channels")
    ax[0].set_xlim([0, 500_000])
    
    # spectrogram
    fs = 10_000 # sampling rate is 10kHz
    f, t, Sxx = signal.spectrogram(train_signal[i], fs)
    ax[1].pcolormesh(t, f, -np.log(Sxx), cmap="plasma")
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_ylim([0, 500])
    ax[1].set_xlabel('Time [sec]')
    plt.tight_layout()
    
    # Power histogram (collapsed across time)
    ax[2].plot(f, np.mean(Sxx, axis=1), color="g")
    ax[2].set_xlabel("Frequency [Hz]")
    ax[2].set_xlim([0, 500])
    ax[2].set_ylabel("Power")
    
spectrogram_plot(train_signal, train_opench, 0)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 1)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 2)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 3)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 4)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 5)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 6)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 7)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 8)


# In[ ]:


spectrogram_plot(train_signal, train_opench, 9)


# The white noise should have equal intensity across different frequencies. According to the spectrograms above, the added noise does not look like the perfect white noise.
# Rather the Fourier space exhibits the sharp increase up to 30-40Hz then gradual decrease as a function of frequency.
