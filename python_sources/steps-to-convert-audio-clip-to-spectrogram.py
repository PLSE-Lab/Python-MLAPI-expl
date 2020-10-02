#!/usr/bin/env python
# coding: utf-8

# The purpose of creating this kernel is to provide - **not only** a step by step guide on how to convert a given audio clip to spectrogram which will be useful for various other audio analysis **but also** to explain what each step in audio loading and visualiztion is doing.<br>
# 
# Provided some links in reference section at the end of the kernel.
# 
# ** More information to be added

# ## Step-1: Let's import all the required libraries

# In[2]:


import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd

audio_fpath = "../input/audio/audio/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))


# ## Some information about audio data before we start with audio data processing
# ### What are x and y axis in a audio wave representation?
# ![Sound wave image](https://swphonetics.files.wordpress.com/2012/03/wavsin01.jpg)
# - The y-axis represents sound pressure, the x-axis represents time.
# 
# ### Standard waveforms
# #### Sine waveform
# ![Sine wave image](https://www.electronics-tutorials.ws/wp-content/uploads/2018/05/waveforms-tim1.gif)
# 
# #### Square waveform
# ![Square waveform image](https://www.electronics-tutorials.ws/wp-content/uploads/2018/05/waveforms-tim3.gif)
# 
# #### Rectangular waveform
# ![Rectangular waveform image](https://www.electronics-tutorials.ws/wp-content/uploads/2018/05/waveforms-tim6.gif)
# 
# #### Triangular waveform
# ![Triangular waveform image](https://www.electronics-tutorials.ws/wp-content/uploads/2018/05/waveforms-tim8.gif)
# 
# #### Sawtooth waveform
# ![Sawtooth waveform image](https://www.electronics-tutorials.ws/wp-content/uploads/2018/05/waveforms-tim9.gif)
# 
# ** More info will be added here

# ## Step-2: Load audio file and visualize its waveform (using librosa)

# In[3]:


x, sr = librosa.load(audio_fpath+audio_clips[2], sr=44100)

print(type(x), type(sr))
print(x.shape, sr)


# In[4]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# ## Step-3: Convert the audio waveform to spectrogram

# In[5]:


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


# ### References:
# - https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
# - https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
# - https://swphonetics.com/praat/tutorials/understanding-waveforms/
# - https://swphonetics.com/praat/tutorials/understanding-waveforms/standard-waveforms/
# - https://www.electronics-tutorials.ws/waveforms/waveforms.html
