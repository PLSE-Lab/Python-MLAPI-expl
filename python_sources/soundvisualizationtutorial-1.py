#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#No need of running this,lol
Notebook details:
  * created on:6.5.2020
  * created by:Ashwani Rathee
  * last updated:6.5.2020
  * Suggestions for improvements and feedback is appreciated.
  * Also ask for update if it stops working,I'll update it
  * UPVOTE!!It helps in maintaining my momentum and faster updates:)


# I am mainly interested in two libraries for this:
# * **Librosa**:a python package for music and audio processing By Brian MacFee.
# And Another Interesting Library is **IPython.display.Audio** that lets you play audio directly in an IPython notebook.
# 
# * Librosa:[Website](http://https://librosa.github.io/)
# * Ipython:[Website](https://ipython.org/ipython-doc/3/api/generated/IPython.display.html)
# * Site I am learning from :[Website](https://musicinformationretrieval.com/)

# In[ ]:


Hi,this is my first tutorial ever.
So, Ill try my best to keep the tutorials short and intuitive for beginners.
My aim is to make a series of kernels, starting from this one, which is a quick approach of loading,understanding by visualization of audio data.
I will gradually add more tutorials on more advanced techniques. 
I hope it is useful, thanks!


# In[ ]:


get_ipython().system('pip install librosa #installing our librosa-the sound director')
get_ipython().system('apt-get install -y ffmpeg libasound2-dev libsndfile1-dev libjack-dev #some other boring packages that help to make it runn well')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt        #for plotting purposes
import librosa.display
import librosa                         #our sound director
import IPython.display as ipd          #our sound director-2


# # Loading the Music:
# *  Misirlou by Dick Dale
# *  Gnossiene-no-1 by Eric Satie

# In[ ]:


#librosa.load to load an audio file into an audio array
x,sr=librosa.load('../input/Misir.mp3') 
ipd.Audio('../input/Misir.mp3')          #ipd.Audio to show the audio in notebook


# In[ ]:


y,sr=librosa.load('../input/Gnossienne-no-1.mp3')
ipd.Audio('../input/Gnossienne-no-1.mp3')


# Printing length of the **audio array** and **sample rate**:

# In[ ]:


print(x.shape)
print(sr)
print(y.shape)
print(sr)


# Visualization of **audio array**:

# In[ ]:


#Misirlou by Dick dale
plt.figure(figsize=(14, 5))                     #setting the size
librosa.display.waveplot(x, sr=sr)              #using librosa.display to plot the graph


# In[ ]:


#Gnossienne-no-1 by Eric satie
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)


# **SPECTROGRAM**:

# In[ ]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')


# In[ ]:


Y = librosa.stft(y)
Ydb = librosa.amplitude_to_db(abs(Y))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Ydb, sr=sr, x_axis='time', y_axis='hz')


# In[ ]:


import numpy
import librosa
sr = 22050 # sample rate
T = 2.0    # seconds
t = numpy.linspace(0, T, int(T*sr), endpoint=False) # time variable
x = 0.5*numpy.sin(2*numpy.pi*440*t)                # pure sine wave at 440 Hz


# In[ ]:


import IPython.display as ipd
ipd.Audio(x, rate=sr) # load a NumPy array


# In[ ]:


#Writting to audio using function
librosa.output.write_wav('audio/tone_440.wav', x, sr)


# Excellent, here we conclude our intro tutorial, but this is just the start!

# If you like my notebook, I would appreciate an upvote, which will keep me motivated for additional content, thanks!
