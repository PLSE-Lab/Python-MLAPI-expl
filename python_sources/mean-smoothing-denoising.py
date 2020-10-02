#!/usr/bin/env python
# coding: utf-8

# ---
# # Mean Smoothing Time Series
# ---

# ### Importing libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
from scipy import *
import copy


# ## Mean Filtering

# **Mean filtering** is a simple method of smoothing and diminishing noise in a signal by eliminating values that are unrepresentative of their surroundings. The idea of mean filtering is to replace each value in a signal point with the mean or average value of its neighbors, including itself.
# 
# $ yt=(2k + 1)^{-1} \sum_{i=t-k}^{t+k} X_i$

# ### Crate a signal

# In[ ]:


srate = 1000 #Hz sample rate -> Number of samples taken by sec
time  = np.arange(0,3,1/srate) 
n     = len(time) 
p     = 15 # poles for random interpolation

# Noise level, measured in standard deviations
noiseamp = 5

# Amp modulator and noise level
ampl   = np.interp(np.linspace(0, p, n), np.arange(0,p), np.random.rand(p)*30)
noise  = noiseamp * np.random.randn(n)
signal = ampl + noise

# Init filter vector with zeros
filtsig = np.zeros(n)


# ### Ploting the noisy signal

# In[ ]:


plt.plot(time, signal, label='noisy')
plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Noise Signal')
plt.show()


# ### Implementing the Mean Smoothing Filter

# In[ ]:


k = 30 # filter window is actually k*2+1
for i in range (k+1, n-k-1):
    filtsig[i] = np.mean(signal[i-k:i+k])

windowsize = 1000*(k*2+1) / srate


# ### Ploting the filtered signal

# In[ ]:


plt.plot(time, signal, label='original')
plt.plot(time, filtsig, label='filtered')

plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Filtered with a k=%d-ms filter Window' %windowsize)
plt.show()

