#!/usr/bin/env python
# coding: utf-8

# ### Overview 
# #### In this competetion Time domain signal is given and task is to predict the number of open channels.
# #### Sampling frequency of the recorded signal is 100KHz, meaning 100,000 cycles in 1 sec of data
# #### The training data consist of 10 recordings, each 50 secs duration
# #### Following YouTube video gives explaination of how the signals are recorded

# In[ ]:


from IPython.display import YouTubeVideo
vid = YouTubeVideo('mVbkSD5FHOw')
display(vid)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sample_sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

print('Train Data')
print('No of train data samples:', len(train.index))
print('No of batches are 10')
display(train.head())
print('Test Data')
print('No of test data samples:', len(test.index))
display(test.head())
print('sample_submission_file')
display(sample_sub.head())


# #### Training Data
# * Following plot displays the 10 batches of the trainng data
# * Each batch has different open channels 
# 

# In[ ]:


train_t = train['time']
train_x = train['signal']
train_ch = train['open_channels']
print('Different values of open channels: ', set(train_ch))

plt.plot(train_t,train_x)
plt.plot(train_t,train_ch,'.')
for i in range(10): plt.plot([train_t[i*500000], train_t[i*500000]],[-5,12.5],'r')
for j in range(10): plt.text(train_t[j*500000+200000],10,str(j+1),size=20)
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')

# What are the open channels in every batch
print('Open channels in every batch:')
for i in range(10):
    print('Batch ', i+1, ':',set(train.loc[0+i*500000:500000+i*500000,'open_channels']))


# * Following plot displays the distribution of signal values for training data

# In[ ]:


import seaborn as sns
sns.set(style="darkgrid")
# analysing first batch data
plt.figure(figsize=(10,10))
ax = sns.countplot(x='open_channels', data=train)


# * Following graphs displays the distribution of signal values for different open channels, for every training data batch

# In[ ]:


import seaborn as sns
sns.set(style="ticks")
# analysing first batch data
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    train_batch = train.loc[0+i*500000:500000+i*500000,['signal','open_channels']]
    ax = sns.countplot(x='open_channels', data=train_batch)
plt.subplots_adjust(hspace=0.9, wspace= 0.9)  


# ### Correlation between Signal Amplitute and Open channels
# * Following graphs shows the signal amplitute for different open channels
# * It is clear that there is no linear relation between signal amplitute and number of open channels

# In[ ]:


plt.plot(train_ch[0:5000000],train_x[0:5000000],'.')
plt.title('Signal Amplitude variation over different open channels')
plt.xlabel('Open Channel #')
plt.ylabel('Signal Amplitude')
plt.xticks(np.arange(0, 11, step=1))
plt.yticks(np.arange(-6, 14, step=2))


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(11):
    train_ch0 = train[train['open_channels']==i]
    train_ch0_t = train_ch0['time']
    train_ch0_x = train_ch0['signal']
    train_ch0_ch = train_ch0['open_channels']
    plt.subplot(4,3,i+1)
    plt.plot(train_ch0_t,train_ch0_x,'.')
    plt.plot(train_ch0_t,train_ch0_ch,'.')
    plt.title('Channel# '+str(i))
    plt.xlabel('Time')
    plt.ylabel('Signal Amplitude')
    plt.yticks(np.arange(-6, 12, step=2))
plt.subplots_adjust(hspace=0.9, wspace= 0.9) 


# ### Batch 5 in the training data has the maximum variations in open channels. Let's examin the Batch 5 data closely.

# * Following plots shows the first 100 samples from the 5th Batch and frequency spectrum for the same

# In[ ]:


fs = 100000
train_x_batch5 = train_x[5*500000:100+5*500000]
train_t_batch5 = train_t[5*500000:100+5*500000]
train_ch_batch5 = train_ch[5*500000:100+5*500000]
plt.plot(train_x_batch5)
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
from scipy import signal
# train_x_batch5_filt = signal.detrend(train_x_batch5)
# plt.subplot(2,1,2)
# plt.plot(train_x_batch5_filt)

sig = train_x_batch5
time_step = 1/fs
period = 5.0
time_vec = train_t_batch5
from scipy import fftpack
# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection


# * Following plot shows the orignal and filtered signals. A low pass filtering is applied.
# * By examining the open channels during this time it is clear that application of any smoothing filter is not appropriate.

# In[ ]:


high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.subplot(2,1,1)
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(train_ch_batch5)
plt.xlabel('Time [s]')
plt.ylabel('Open Channels')


# ## Analyzing the test data

# * The test data consist of 4 recordings, each 50 secs duration

# In[ ]:


test_t = test['time']
test_x = test['signal']
# test_ch = test['open_channels']
# print('Different values of open channels: ', set(train_ch))

plt.plot(test_t,test_x)
# plt.plot(test_t,test_ch,'.')
for i in range(4): plt.plot([test_t[i*500000], test_t[i*500000]],[-5,12.5],'r')
for j in range(4): plt.text(test_t[j*500000+200000],4,str(j+1),size=20)
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')

