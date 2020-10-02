#!/usr/bin/env python
# coding: utf-8

# In the case of phase-sensitive multi-microphone neural network models, there is a simple augmentation called [Spectral Distortion Model](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/78ea22a4a1cdfda4103237812fc92b9642b52109.pdf).
# 
# I assume that this type of augmentation should also work in this [challenge](https://www.kaggle.com/c/dcase2019-task1b-leaderboard) with a different type of devices.
# 
# In this kernel, I will demonstrate how to apply the phase distortion augmentation.

# In[ ]:


import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt


# Load original signal

# In[ ]:


x, sr = librosa.load('../input/freesound-audio-tagging-2019/train_curated/f5342540.wav', sr=44100)
x = x[:len(x) // 7]


# Create a random phase distortion model with sigma equal to 0.4

# In[ ]:


np.random.seed(2019)
PDM = np.array([np.complex(np.cos(p), np.sin(p)) for p in np.random.normal(0, 0.4, size=513)])


# Transform the original signal and apply augmentation for each frame

# In[ ]:


f = librosa.stft(x, window='hanning', hop_length=512, n_fft=1024)
f *= PDM.reshape((-1, 1))


# Reconstruct transformed signal

# In[ ]:


y = librosa.istft(f, window='hanning', hop_length=512)
x = x[:len(y)]


# In the figure below, you can see a small difference between the two signals

# In[ ]:


plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(x)

plt.subplot(2, 1, 2)
plt.plot(y)

plt.show()


# Signals sound equally

# In[ ]:


ipd.Audio(x, rate=sr)


# In[ ]:


ipd.Audio(y, rate=sr)


# According to mention paper, this augmentation also worked with Multistyle-TRaining (MTR).
# 
# It would be interesting to test this approach for the acoustic scene classification or event detection tasks.
