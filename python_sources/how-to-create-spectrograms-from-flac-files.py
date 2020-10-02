#!/usr/bin/env python
# coding: utf-8

# This notebook shows a quick example of how to read FLAC files into numpy arrays and draw spectorgrams from them. 
# 
# Here are the packages we'll need:

# In[7]:


import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt


# Now, let's read in a sample birdsong file. 

# In[8]:


data, samplerate = sf.read("../input/songs/songs/xc101862.flac")


# I'm going to show you two functions for drawing spectrograms. The first approach is to use the `scipy.signal.spectogram()` function to calculate the spectrogram, and then plot it using matplotlib: 

# In[9]:


freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')
plt.pcolormesh(time, freq, Sxx)

# add axis labels
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


# I find that the defaults on `pcolomesh()` don't work great for most speech/bioacoustics data. Another option that I generally have more luck with is to use the `specgram()` function from pyplot: 

# In[10]:


Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)

# add axis labels
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


# And that's all there is to it! :)
