#!/usr/bin/env python
# coding: utf-8

# # An Intuitive Application of Butterworth Filters
# 
# From [Tomas' PhD thesis ](http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf), the following diagram is given (on pg. 30) to show steps taken to de-noise the signals:
# ![UML](https://i.imgur.com/No45Kad.jpg)

# ### This kernel will focus on the first two steps here - sine wave synchronisation and sine suppression.
# 
# The method of removing the sine component from the signals outlined in the thesis involves taking a low pass filter with a cut-off frequency of 50Hz and subtracting the filtered signal from the raw synchronised sine wave to leave only the high-frequency components - including the PD patterns. Synchronisation is carried out to allow easy comparison between signals that are out of phase. The resulting signal is then de-noised using DWT.
# 
# So what does this mean?
# 
# Well, from our data description, we know that each signal contains 800,000 measurements of a power line's voltage, taken over 20 milliseconds and the underlying electric grid operates at 50 Hz. So each signal covers one complete grid cycle. This means that the sinusoidal components of the signals we are given have a frequency of 50Hz. The higher frequency components are due to noise from radio emissions (DSI), interference from electronics (RPI), and other sources. However, the partial discharge patterns we're looking for also make up these high frequency components. Therefore, if we use a low-pass filter with a cut-off frequency of 50Hz, we should maintain the sine wave while removing the noise in the signals. Then we can simply subtract the clean sine wave from our raw signal to leave us with the high-frequency components we're looking for. A great diagram of this is given in the aforementioned thesis:
# 
# ![filtering](https://i.imgur.com/WhYchFG.jpg)
# 
# Let's try implementing this.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Read in the first three signals. We know from our EDA that these do not possess any PD faults.

# In[ ]:


train_sig = pd.read_parquet('../input/train.parquet', columns=[str(i) for i in range(3)])


# In[ ]:


train_sig.head()


# In[ ]:


train_sig.shape


# Let's first synchronise the signals. Here's what the first three signals in the training data look like:

# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
for i in range(3):
    sns.lineplot(train_sig.index, train_sig[str(i)])


# We know that each of the three signals are 120 degrees out of phase with each other. One of the signals looks to be begining its cycle at the first measurement point (x=0). Let's get the other signals to synchronise with this one. We can see that one of the signals crosses the x-axis from negative amplitude around x=525,000 and the other does the same at about x=275,000. This holds true for each trio of signals in our dataset due to the fundamental nature of [three-phase electric power](https://en.wikipedia.org/wiki/Three-phase_electric_power).
# 
# To synchronise the signals, let's split them at the identified x-axis crossing points and concatenate the left side of the signal with the right side.

# In[ ]:


train_sig['0'][550000:].values


# In[ ]:


train_sig['0'][:550000].values


# In[ ]:


train_sig['0']=np.concatenate([train_sig['0'][525000:].values, train_sig['0'][:525000].values])
train_sig['2']=np.concatenate([train_sig['2'][275000:].values, train_sig['2'][:275000].values])


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
for i in range(3):
    sns.lineplot(train_sig[:800000].index, train_sig[str(i)][:800000])


# Look's a bit off but since we only approximated the points at which the signals crossed the x-axis, as well their noisy nature, it's good enough for now.
# 
# Now let's filter out the higher-frequency components of the signals.

# In[ ]:


from scipy.signal import butter, lfilter


# Recall from signal processing:
# 
# Nyquist frequency = 1/2 * (sampling rate) Hz
# 
# We know from our data description that our sampling rate = 800,000 measurements / 20*(10**-3) seconds = 40MHz
# 
# Therefore, our Nyquist frequency is 20MHz.
# 
# For digital filters in [scipy.signal.butter](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html) (SciPy's implementation of the butterworth filter), Wn is normalized from 0 to 1 where 1 is the Nyquist frequency.
# 
# So for a cutoff of 50Hz, Wn = 50Hz / 20MHz = 2.5 * (10**-6)
# 
# Let's see how this looks coded up.

# In[ ]:


cutoff=50


# In[ ]:


measurements=800000


# In[ ]:


time=0.02


# In[ ]:


sampling_rate = measurements/time


# In[ ]:


sampling_rate


# In[ ]:


cutoff/(sampling_rate*0.5)


# In[ ]:


nyquist = sampling_rate*0.5


# In[ ]:


wn = cutoff/nyquist


# Implementing lowpass filter on the first signal (signal_id = 0):

# In[ ]:


b, a = butter(3, wn, btype='lowpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)


# In[ ]:


filtered_sig.shape


# In[ ]:


filtered_sig


# In[ ]:


train_sig['3'] = filtered_sig


# In[ ]:


train_sig.head(20)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['3'][:800000])


# We get a pretty clean sinusoidal signal after filtering but it looks like we've gotten some phase shift here. Let's try subtracting this from our raw synchronised signal and see what happens.

# In[ ]:


train_sig['4'] = train_sig['0'] - train_sig['3']


# In[ ]:


train_sig.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['4'][:800000])


# Oof. This definitely doesn't look anything like the flattened signal in Tomas' thesis. Looks like we're getting some constructive interference here - the timing and phase alignment aren't quite good enough for the destructive interference we're looking for. 
# 
# So what now?
# 
# What if we instead filtered out the low frequency component (the sine wave) with a high-pass filter mainting everything in our signal above 50Hz?

# In[ ]:


b, a = butter(3, wn, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)


# In[ ]:


filtered_sig.shape


# In[ ]:


filtered_sig


# In[ ]:


train_sig['5'] = filtered_sig


# In[ ]:


train_sig.head(20)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['5'][:800000])


# Still not really getting rid of the sinusoidal component. Let's try upping the cutoff frequency to 5000Hz instead since the PD patterns have a much greater frequency than anything else we'd be looking for in the signal. This should hopefully get rid of the sine wave but still maintain the partial discharge.

# In[ ]:


b, a = butter(3, 5000/nyquist, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)


# In[ ]:


filtered_sig.shape


# In[ ]:


filtered_sig


# In[ ]:


train_sig['6'] = filtered_sig


# In[ ]:


train_sig.head(20)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['6'][:800000])


# This looks much more reasonable. But maybe 5000Hz is too high of a cut-off frequency? We might be losing some important PD information here. Let's keep reducing the cut-off until we start seeing the sinusoidal component again and choose a value just above this.

# ## 2500Hz cut-off freq.

# In[ ]:


b, a = butter(3, 2500/nyquist, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)


# In[ ]:


filtered_sig.shape


# In[ ]:


filtered_sig


# In[ ]:


train_sig['7'] = filtered_sig


# In[ ]:


train_sig.head(20)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['7'][:800000])


# ## 1000Hz cut-off freq.

# In[ ]:


b, a = butter(3, 1000/nyquist, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)


# In[ ]:


filtered_sig.shape


# In[ ]:


filtered_sig


# In[ ]:


train_sig['8'] = filtered_sig


# In[ ]:


train_sig.head(20)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['8'][:800000])


# Looks some sinusoidal stuff is creeping back in here. Somewhere around 2.5kHz to 5kHz could be a good cut-off.
