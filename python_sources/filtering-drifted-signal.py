#!/usr/bin/env python
# coding: utf-8

# In this notebook I review some filtering technique to remove the drifted signal. The idea is conceived when I came across this notebook https://www.kaggle.com/cdeotte/one-feature-model-0-930 .The author manually model the drifted signal (take some points from the graph then fit a polinomial) and it is very natural in this case because the signal is just 1st and 2nd order. But how about in a real case when it is not that evident.
# 
# In signal processing, people normally try to use a high pass filter to remove this low frequency signal

# In[ ]:


import numpy as np 
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift


# In[ ]:


train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sample_sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


train_df.head()


# The original signal is plotted as below. You can easily recognize the drifted signal.

# In[ ]:


plt.figure(figsize=(20,5)); res=1000
plt.plot(train_df.time[::res], train_df.signal[::res])
plt.ylabel('signal')
plt.xlabel('time')
plt.show()


# Use a highpass Butterworth signal then use can get a filtered signal without the drift.

# In[ ]:


sos = signal.butter(5, 10, 'hp', fs=10000, output='sos')
filtered = signal.sosfilt(sos, train_df.signal)


# In[ ]:


plt.figure(figsize=(20,5)); res=1000
plt.plot(train_df.time[::res], filtered[::res])


# In the next update, I will try to carrefully choose the cut off frequency by reviewing the frequency spectrum of the parabolic signal. I also need to retain the mean value of signal during each batch because it is removed after filtering.
