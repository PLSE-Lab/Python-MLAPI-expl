#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# Since there are not a lot of data in our trainset, let us create some more. The basic idea here is simple: a guitar + another guitar = still a guitar. What I will try is to add the audio signal and check the result.
# 
# This kernel is still a work in progress, so far I was lucky to have no values out of range of the 16bits audio file, but I will have to check for this in later versions.

# # 2. Mixing Two Audio Files
# ## 2.1 A First Try on Two Files
# 
# Let us take two audio files and add there signal. Here, I take two different sounds (a trumpet and a cello).
# 

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels = pd.read_csv('../input/train.csv')
labels[labels['label'] == 'Trumpet'].head()


# In[10]:


labels[labels['label'] == 'Cello'].head()


# Let us listen to this separately.

# In[19]:


trumpet = '034e4ffa'
cello = '00353774'


# In[25]:


import IPython.display as ipd  # To play sound in the notebook
t_fname = '../input/audio_train/' + trumpet + '.wav'
ipd.Audio(t_fname)


# In[26]:


import IPython.display as ipd  # To play sound in the notebook
c_fname = '../input/audio_train/' + cello + '.wav'
ipd.Audio(c_fname)


# It is now time to add signals.

# In[47]:


from scipy.io import wavfile
rate, t_signal = wavfile.read(t_fname)
rate, c_signal = wavfile.read(c_fname)

min_len = min(len(t_signal),len(c_signal))

t_signal = np.array([(e/2**16.0)*2 for e in t_signal]) #16 bits tracks, normalization
c_signal = np.array([(e/2**16.0)*2 for e in c_signal])

t_signal = t_signal[:min_len]
c_signal = c_signal[:min_len]

new_sig = t_signal + c_signal

new_sig_16b = np.array([int((v*2**16.0)/2) for v in new_sig])
plt.plot(new_sig_16b)

ipd.Audio(new_sig_16b,rate=44100)


# I have heard more harmonious things in my life, but still we here distincly a cello and a trumpet. Let us try with two acoustic guitars.
# 
# ## 2.3 Same Example with Twice the Same Label

# In[52]:


labels[labels['label'] == 'Acoustic_guitar'].head()


# In[54]:


g1 = '0356dec7'
g1_fname = '../input/audio_train/' + g1 + '.wav'
ipd.Audio(g1_fname)


# In[59]:


g2 = '0969b5c5'
g2_fname = '../input/audio_train/' + g2 + '.wav'
ipd.Audio(g2_fname)


# In[77]:


rate, g1_signal = wavfile.read(g1_fname)
rate, g2_signal = wavfile.read(g2_fname)

min_len = min(len(g1_signal),len(g2_signal))

g1_signal = np.array([(e/2**16.0)*2 for e in g1_signal]) #16 bits tracks, normalization
g2_signal = np.array([(e/2**16.0)*2 for e in g2_signal])

g1_signal = g1_signal[:min_len]
g2_signal = g2_signal[:min_len]

new_sig = g1_signal + g2_signal

new_sig_16b = np.array([int((v*2**16.0)/2) for v in new_sig])
g1_sig_16b = np.array([int((v*2**16.0)/2) for v in g1_signal])
g2_sig_16b = np.array([int((v*2**16.0)/2) for v in g2_signal])
plt.plot(new_sig_16b)

ipd.Audio(new_sig_16b,rate=44100)


# I won't speculate on the accord we got here, but to me, we clearly have two guitars. Let's check this on a spectrogram.

# In[89]:




from scipy import signal
#data
freqs, times, specs = signal.spectrogram(new_sig,
                                         fs=44100,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')
freqs, times, specs1 = signal.spectrogram(g1_signal,
                                         fs=44100,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')
freqs, times, specs2 = signal.spectrogram(g2_signal,
                                         fs=44100,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')

max_freq = 1000
plt.plot(freqs[:max_freq],np.absolute(specs[:max_freq,1]),'r')
plt.plot(freqs[:max_freq],np.absolute(specs1[:max_freq,1]),'bo')
plt.plot(freqs[:max_freq],np.absolute(specs2[:max_freq,1]),'gx')


# The spectrum of the two signals in red is clearly the mix of the individual signals (green and blue). It seems that we have a new datapoint here.
# 
# Now, let us wrap a function to generate infinitely many new audio signals.
# 
# # Wrapper
# 
# WIP
