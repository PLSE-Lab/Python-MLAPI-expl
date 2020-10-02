#!/usr/bin/env python
# coding: utf-8

# ### Voice activity detection example
# 
# in (https://www.kaggle.com/davids1992/data-visualization-and-investigation)[this kernel] it was proposed to shrink the samples to those parts where speech could be identified (to speed up training) using webrtcvad
# 
# (note that this will not work on kaggle.com as webrtcvad is not available there)

# In[1]:


import os
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# directory with input data

# In[2]:


train_audio_path = "../input/train/audio"


# example input file (from the same kernel mentioned above)

# In[3]:


filename = 'yes/0a7c2a8d_nohash_0.wav'


# read the sound file

# In[4]:


from scipy.io import wavfile
sample_rate, samples = wavfile.read(os.path.join(train_audio_path, filename))


# use the webrtcvad library to identify segments as speech or not

# In[5]:


import webrtcvad


# In[6]:


vad = webrtcvad.Vad()

# set aggressiveness from 0 to 3
vad.set_mode(3)


# convert samples to raw 16 bit per sample stream needed by webrtcvad

# In[7]:


import struct
raw_samples = struct.pack("%dh" % len(samples), *samples)


# run the detector on windows of 30 ms 
# (from https://github.com/wiseman/py-webrtcvad/blob/master/example.py)

# In[8]:


window_duration = 0.03 # duration in seconds

samples_per_window = int(window_duration * sample_rate + 0.5)

bytes_per_sample = 2


# In[9]:


segments = []

for start in np.arange(0, len(samples), samples_per_window):
    stop = min(start + samples_per_window, len(samples))
    
    is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
                              sample_rate = sample_rate)

    segments.append(dict(
       start = start,
       stop = stop,
       is_speech = is_speech))
    
    


# plot the range of samples identified as speech in orange

# In[10]:


plt.figure(figsize = (10,7))
plt.plot(samples)

ymax = max(samples)

# plot segment identifed as speech
for segment in segments:
    if segment['is_speech']:
        plt.plot([ segment['start'], segment['stop'] - 1], [ymax * 1.1, ymax * 1.1], color = 'orange')

plt.xlabel('sample')
plt.grid()


# listen to the speech only segments

# In[11]:


speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])

import IPython.display as ipd
ipd.Audio(speech_samples, rate=sample_rate)


# In[ ]:




