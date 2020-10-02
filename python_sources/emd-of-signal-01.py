#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# I'm going to depart from the usual procedure of extracting statistical measures of the signal and then creating features from them. Instead, I'm going to apply a method called **Empirical Mode Decomposition** also known as the **Hilbert-Huang Transform**. This is a method which has been already applied to geophysical signals (insert reference here) and that is chiefly applied to non-linear, non-stationary signals. Seismic signals usually fall on this category.
# 
# ## **Empirical Mode Decomposition**
# The Empirical Mode Decomposition (EMD) method is way of analyzing a non-linear, non-stationary signal by decomposing it into a series of zero-mean, oscillating components known as **intrinsic mode functions** (IMFs). The instantaneous frequency of these components is then obtained by applying the Hilbert transform to get a physically meaningful characterization of the signal. 
# The EMD method was introduced in 1998 by Norden Huang and collaborators as a method of analyzing various geophysical signals obtained in the course of their investigations.
# 
# ## **Library PyEMD**
# The library used to compute the EMD of the 16 earthquake signals was the excellent PyEMD (*Dawid Laszuk (2017-), Python implementation of Empirical Mode Decomposition algorithm. http://www.laszukdawid.com/codes.*). This library is the Python implementation of a previously implemented MATLAB code. Thanks very much to Dawid Laszuk for making this available.
# 
# ## **Approach**
# The tactic I'm taking is to compute the first 20 IMFs of each signal, compute the usual statistical subjects, and then use these as features for each signal. Let's see what happens.

# In[3]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from scipy.signal import hilbert
from PyEMD import EMD
pd.options.display.precision = 10
from os import listdir
print(listdir("../input"))


# In[ ]:


def instant_phase(imfs):
    """Extract analytical signal through Hilbert Transform."""
    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    # Compute angle between img and real
    phase = np.unwrap(np.angle(analytic_signal))
    return phase


# In[ ]:


signal = pd.read_csv('../input/Signal01.csv')
print(signal.head())
S = signal.signal.values[::10]
t = signal.quaketime.values[::10]
print('S shape: ', S.shape)
print('t shape: ', t.shape)


# In[ ]:


dt = t[0] - t[1]
print(dt)


# I've configured the EMD class with the parameters seen below to save some computing time. 

# In[ ]:


# Compute IMFs with EMD
config = {'spline_kind':'linear', 'MAX_ITERATION':100}
emd = EMD(**config)
imfs = emd(S, max_imf=10)
print('imfs = ' + f'{imfs.shape[0]:4d}')


# In[ ]:


# Extract instantaneous phases and frequencies using Hilbert transform
instant_phases = instant_phase(imfs)
instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)


# In[ ]:


# Create a figure consisting of 3 panels which from the top are the input 
# signal, IMFs and instantaneous frequencies
fig, axes = plt.subplots(3, figsize=(12, 12))

# The top panel shows the input signal
ax = axes[0]
ax.plot(t, S)
ax.set_ylabel("Amplitude [arbitrary units]")
ax.set_title("Input signal")

# The middle panel shows all IMFs
ax = axes[1]
for num, imf in enumerate(imfs):
    ax.plot(t, imf, label='IMF %s' %( num + 1 ))

# Label the figure
#ax.legend()
ax.set_ylabel("Amplitude [arb. u.]")
ax.set_title("IMFs")

# The bottom panel shows all instantaneous frequencies
ax = axes[2]
for num, instant_freq in enumerate(instant_freqs):
    ax.plot(t[:-1], instant_freq, label='IMF %s'%(num+1))

# Label the figure
#ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Inst. Freq. [Hz]")
ax.set_title("Huang-Hilbert Transform")

plt.tight_layout()
plt.savefig('Signal-01-Amplitudes', dpi=120)
plt.show()


# In[ ]:


# Plot results
nIMFs = imfs.shape[0]
plt.figure(figsize=(24,24))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(S, 'r')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(imfs[n], 'g')
    plt.ylabel("IMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
#plt.tight_layout()
plt.savefig('Signal-01', dpi=120)
plt.show()


# In[ ]:


# The top panel shows the input signal
ax = axes[0]
ax.plot(S)
ax.set_ylabel("Amplitude [arbitrary units]")
ax.set_title("Input signal")
plt.show()


# In[ ]:


plt.plot(S)
plt.show()


# In[ ]:




