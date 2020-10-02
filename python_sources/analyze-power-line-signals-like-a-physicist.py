#!/usr/bin/env python
# coding: utf-8

# # Analyze Power Line Signal Like a Physicist
# We read and explore the data, especially the labels. Then, we focus on the frequency domain. Digital filtering with infinitely fast roll-off (sharp cutoff) is demonstrated. 
# Statistician likes to use moving average and other smoothing techniques, which are basically low-pass filters with very slow roll-off. Engineers prefer more realistic filters with finite roll-off, because they have to implement filter in the real world. That is why scipy.signal provides an array of different filters. If you are a physicist who doesn't care about real world, why don't we just filter with infinitely fast roll-off?  
# Before we begin, I would like to thank https://www.kaggle.com/xhlulu/exploring-signal-processing-with-scipy and the host: https://www.kaggle.com/sohier/reading-the-data-with-python

# In[ ]:


import numpy as np
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq


# ## Read and Explore
# Read signals (variable x). 

# In[ ]:


xs = pq.read_table('../input/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
# xs = pq.read_table('../input/train.parquet').to_pandas()
print((xs.shape))
xs.head(2)

Read labels (variable y). 
# In[ ]:


train_meta = pd.read_csv('../input/metadata_train.csv')
print(train_meta.shape)
train_meta.head(6)
train_meta_good = train_meta[train_meta.target == 0]
train_meta_error = train_meta[train_meta.target == 1]
id_error = train_meta_error.groupby('id_measurement')['target'].count()
id_error.head(6)


# Examine how many phases with one ID were labelled problematic. For 80.4% faulty lines, three phases were all labelled faulty, while one-faulty-phase and two-faulty-phase lines contribute about 10% and 10%, respectively.  

# In[ ]:


id_error_c = id_error.astype('category')
print(id_error_c.value_counts())
print(id_error_c.value_counts() / id_error_c.value_counts().sum())
id_error_c.value_counts().plot(kind='bar')


# In[ ]:


period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print(f'Sampling Frequency = {f_sampling / 1e6} MHz')
# print (str(50* 800000 /1e6) + ' MHz')


# In[ ]:


# Fetch one signal from xs
idx = 1
sig = xs.iloc[:, idx]
idx_error = 3
sig_error = xs.iloc[:, idx_error]
print(sig.shape)


# In[ ]:


# https://www.scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
# The FFT of the signal
sig_fft = fftpack.fft(sig)
# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)
# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq >= 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

plt.figure(figsize=(6, 5))
# plt.plot(sample_freq[pos_mask], power[pos_mask])
plt.semilogy(sample_freq[pos_mask], power[pos_mask])
plt.ylim([1e-0, 1e8])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [A.U./Hz]')


# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.6, 0.3, 0.2])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])


# The inset figure above shows that peak frequency is 50 Hz, as we expected. Note that the frequency step in fft spectrum is 50 Hz, limited by the total duration of the signal. 
# Next we use a slightly different method (signal.periodogram) and plot the power spectrum in log-log scale, which make more sense. The unit in the y axis is different by a fixed factor. It is OK. 

# In[ ]:


def plot_ps(sig, f_sampling, label='sig', style='loglog'):
    f, Pxx_den = signal.periodogram(sig, f_sampling)
    if style == 'semilogy':
        plt.semilogy(f, Pxx_den, label=label)
    else:
        plt.loglog(f, Pxx_den, label=label)
    plt.ylim([1e-9, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [A.U./Hz]')
    
plot_ps(sig, f_sampling, 'Good sig')
plot_ps(sig_error, f_sampling, 'Bad sig')
plt.legend(loc='best')
plt.show()
# The horizontal line at 50 Hz is artificial, as the log scale in the x axis cannot show 0 Hz. 


# In[ ]:


def bandpassfilter(spec, sample_freq, lowcut, highcut):
    # a digital bandpass filter with a infinite roll off. 
    # Note that we will keep the frequency point right at low cut-off and high cut-off frequencies. 
    spec1 = spec.copy()
    spec1[np.abs(sample_freq) < lowcut] = 0
    spec1[np.abs(sample_freq) > highcut] = 0
    filtered_sig = fftpack.ifft(spec1)
    return filtered_sig


# ## Digital filtering
# The peak_freq should be 50 Hz. 
# We demonstrated differnt low-pass, high-pass, and band-pass filtered signals. You can see 10-1000 Hz can capture a lot of low frequency features. 
# 

# In[ ]:


# We demonstrated differnt low-pass filtered signals. You can see 10-1000 Hz can capture a lot of low frequency features. 
lowcut, highcut = 10, 100
filtered_sig0 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 10, 300
filtered_sig1 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 10, 1000
filtered_sig2 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig0, linewidth=3, label='10-100 Hz')
plt.plot(time_vec, filtered_sig1, linewidth=3, label='10-300 Hz')
plt.plot(time_vec, filtered_sig2, linewidth=3, label='10-1000 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')

# We also demonstrate a band-pass filtered and a high-pass filtered signals. 
lowcut, highcut = 1000, 1e6
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 40e6
filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig4, linewidth=3, label='Above 1 kHz')
plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')

