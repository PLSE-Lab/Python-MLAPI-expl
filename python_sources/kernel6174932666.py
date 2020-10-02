#!/usr/bin/env python
# coding: utf-8

# In[53]:


import os,librosa.display
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.listdir('../input')
d=pd.read_csv('../input/Knjiga1.csv',delimiter=';')


# In[16]:


fun1=d['F1']
fun2=d['F2']
f1=list(fun1)
f2=list(fun2)
for i,j,k,n in zip(f1,f2,['b','green','c','k','m'],d['Vowel']):
    plt.scatter(i,j,color=k,label=n)
plt.legend()


# In[17]:


import IPython.display as ipd
import librosa as lbr
x,sr=lbr.load('../input/dorianbeli.wav')
ipd.Audio(x,rate=sr)


# In[35]:


plt.figure(figsize=(17,8))
librosa.display.waveplot(x, sr)


# In[39]:


r = librosa.autocorrelate(x, max_size=7000)
plt.figure(figsize=(14, 5))
plt.plot(r[:300])


# In[55]:


import wave
with  wave.open('../input/dorianbeli.wav') as wav:
    frames=wav.readframes(-1)
    rate=wav.getframerate()
    info=np.fromstring(frames,'int16')
plt.specgram(info,Fs=rate)

