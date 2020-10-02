#!/usr/bin/env python
# coding: utf-8

# ## Step-1: Let's import all the required libraries

# In[ ]:


import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd

audio_fpath = "../input/audio/audio/16000/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))


# ## Step-2: Load audio file and visualize its waveform (using librosa)

# In[ ]:


x, sr = librosa.load(audio_fpath+audio_clips[2], sr=16000)

print(type(x), type(sr))
print(x.shape, sr)


# In[ ]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# ## Step-3: Convert the audio waveform to spectrogram

# In[ ]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# ## Step-4: Applying log transformation on the loaded audio signals

# In[ ]:


plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

