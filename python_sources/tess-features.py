#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports
import librosa
import librosa.display
import librosa.feature
import numpy 
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import os
import IPython.display as ipd  # To play sound in the notebook


# We Find the following things and then maybe use them in future to predict emotions I believe..
# - Root Mean Square Energy
# - Zero Crossing Rate,Spectrogram,Mel Spectrogram
# - Spectral Centroid
# - Spectral Bandwidth
# - Spectral Contrast
# - Spectral Rolloff
# - chroma vector
# - Constant-Q Transform
# - MFCC

# In[ ]:


#We focus on Only 1 audio as of Now and that is ../input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav


# In[ ]:


x, sr = librosa.load('../input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav')


# In[ ]:


x.shape#array length


# In[ ]:


sr#sample_Rate


# In[ ]:


librosa.get_duration(x) #length of audio


# In[ ]:


ipd.Audio('../input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav') # load a local WAV file


# In[ ]:


plt.figure(figsize=(25,7))
librosa.display.waveplot(x, sr=sr,x_axis='time',offset=0)#max sample rate and maxpoints to plot can be specified too
plt.title('Mono')


# In[ ]:


#Root mean square value
rms=librosa.feature.rms(y=x)
rms


# In[ ]:


rms.shape#so this has 75 points for 1.712 length of audio


# In[ ]:


#Zero crossing rate
zcr=librosa.feature.zero_crossing_rate(x)
zcr


# In[ ]:


zcr.shape#(this also has 75 points of data)


# In[ ]:


chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr,n_chroma=12, n_fft=4096)
chroma_stft.shape


# In[ ]:


chroma_stft[0]
#chroma_stft[1]
#chroma_stft[2]


# In[ ]:


chroma_cqt = librosa.feature.chroma_cqt(y=x, sr=sr, n_chroma=12)
chroma_cqt 
#this also can produce those 12 features or n=more


# In[ ]:


#13 mfcc features
mfcc_full=librosa.feature.mfcc(y=x, sr=sr,n_mfcc=12)


# In[ ]:


mfcc_full.shape


# In[ ]:


mfcc_full[0] #first 75 features like wise can be done for all other
#mfcc_Full[1]
#mfcc_full[2]


# In[ ]:


#spectral_Centroid
spectral_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
spectral_cent#it has 75 values too


# In[ ]:


spectral_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
spectral_bw


# In[ ]:


spectral_flatness = librosa.feature.spectral_flatness(y=x)
spectral_flatness


# In[ ]:


spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
spectral_rolloff


# In[ ]:


import csv


# In[ ]:


pip install spafe


# In[ ]:


import scipy
from spafe.utils import vis
from spafe.features.gfcc import gfcc


# In[ ]:


pwd()


# In[ ]:


# init input vars
num_ceps = 13
low_freq = 0
high_freq = 2000
nfilts = 24
nfft = 512
dct_type = 2,
use_energy = False,
lifter = 5
normalize = False


# In[ ]:


# read wav 
fs, sig = scipy.io.wavfile.read("../input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav")


# In[ ]:


# compute features
gfccs = gfcc(sig=x,
             fs=sr,
             num_ceps=num_ceps,
             nfilts=nfilts,
             nfft=nfft,
             low_freq=low_freq,
             high_freq=high_freq,
             dct_type=dct_type,
             use_energy=use_energy,
             lifter=lifter,
             normalize=normalize)

# visualize spectogram
vis.spectogram(sig, fs)
# visualize features
vis.visualize_features(gfccs, 'GFCC Index', 'Frame Index')


# In[ ]:




