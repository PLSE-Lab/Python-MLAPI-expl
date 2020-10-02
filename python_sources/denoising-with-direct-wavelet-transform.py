#!/usr/bin/env python
# coding: utf-8

# # Denoising with the Direct Wavelet Transform
# 
# As stated in the data description :
# > The data is simulated and injected with real world noise to emulate what scientists observe in laboratory experiments.
# 
# So we might want to remove this noise to make the task easier. To do this, I use the Direct Wavelet Transform.
# This kernel continues [the one using the FFT](https://www.kaggle.com/theoviel/denoising-with-the-fast-fourier-transform), but wavelets perform a bit better in my experiments. 
# Although I prefer starting with the FFT usually, because I understand its principle better.
# 
# 
# *Sources : https://www.kaggle.com/theoviel/denoising-with-the-fast-fourier-transform, https://www.kaggle.com/jackvial/dwt-signal-denoising*

# In[ ]:


import pywt
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# In[ ]:


df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv')
df_train.head()


# In[ ]:


n_times = 1000
time = df_train['time'][:n_times].values
signal = df_train['signal'][:n_times].values


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(time, signal)
plt.title('Signal', size=15)
plt.show()


# # Discrete Wavelet Transform (dwt) denoising
# 
# > For the maths behind the dwt, see https://en.wikipedia.org/wiki/Wavelet
# 
# ### Denoising algorithm
# The denoising steps are the following :
# 
# - Apply the dwt to the signal
# - Compute the threshold corresponding to the chosen level
# - Only keep coefficients with a value higher than the threshold
# - Apply the inverse dwt to retrieve the signal

# In[ ]:


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


# In[ ]:


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


# ### Which wavelet to use ?
# We take a look at the available wavelets. The pywt package actually has 127 of them

# In[ ]:


for wav in pywt.wavelist():
    try:
        filtered = wavelet_denoising(signal, wavelet=wav, level=1)
    except:
        pass
    
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Raw')
    plt.plot(filtered, label='Filtered')
    plt.legend()
    plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
    plt.show()


# **Thanks for reading !**
# 
# I hope this can be somehow useful.
