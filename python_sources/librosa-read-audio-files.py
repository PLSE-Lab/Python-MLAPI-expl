#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


# In[ ]:


get_ipython().system('pip install librosa')


# In[ ]:


import librosa as lr


# In[ ]:


ls ../input/


# In[ ]:


path = '../input/'
df_1 = pd.read_csv(path+'set_a.csv')


# In[ ]:


df_1.head()


# In[ ]:


data_dir = '../input/set_a'
audio_files = glob(data_dir+'/*.wav')


# In[ ]:


len(audio_files)


# In[ ]:


audio, sfreq = lr.load(audio_files[0])
time = np.arange(0,len(audio))/sfreq


# In[ ]:


time


# In[ ]:


fig, ax = plt.subplots()
ax.plot(time,audio)


# In[ ]:


audio, sfreq = lr.load(audio_files[1])
time = np.arange(0, len(audio))/sfreq


# In[ ]:


plt.plot(time,audio)


# In[ ]:


for file in range(0, len(audio_files),1):
    audio, sfreq = lr.load(audio_files[file])
    fig,ax = plt.subplots()
    time = np.arange(0,len(audio))/sfreq
    ax.plot(time,audio)
    plt.show()


# In[ ]:




