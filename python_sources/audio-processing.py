#!/usr/bin/env python
# coding: utf-8

# ## Data dowload & exploration

# We begin by creating directory where we'll download our data.

# In[ ]:


DATA_DIR = 'data'
get_ipython().system('mkdir {DATA_DIR} -p')


# Next, let's download and unzip the data:

# In[ ]:


# Only on Linux, Mac and Windows WSL
get_ipython().system('wget http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz')
get_ipython().system('tar -C {DATA_DIR} -zxf ST-AEDS-20180100_1-OS.tgz')
get_ipython().system('rm ST-AEDS-20180100_1-OS.tgz')


# We can listen to audio files directly within Jupyter using a display widget.

# In[ ]:


import os
from IPython.display import Audio

audio_files = os.listdir(DATA_DIR)
len(audio_files), audio_files[:10]


# In[ ]:


example = DATA_DIR + "/" + audio_files[0]
Audio(example)


# In[ ]:


Audio(DATA_DIR + "/" + audio_files[1])


# In[ ]:


Audio(DATA_DIR + "/" + audio_files[823])


# ## Audio signals & sampling
# 
# We'll use the library `librosa` to process and play around with audio files.

# In[ ]:


import librosa


# In[ ]:


y, sr = librosa.load(example, sr=None)


# In[ ]:


print("Sample rate  :", sr)
print("Signal Length:", len(y))
print("Duration     :", len(y)/sr, "seconds")


# Audio is a continuous wave that is "sampled" by measuring the amplitude of the wave at a given time. How many times you sample per second is called the "sample rate" and can be thought of as the resolution of the audio. The higher the sample rate, the closer our discrete digital representation will be to the true continuous sound wave. Sample rates generally range from 8000-44100 but can go higher or lower.

# Our signal is just a numpy array with the amplitude of the wave.

# In[ ]:


print("Type  :", type(y))
print("Signal: ", y)
print("Shape :", y.shape)


# We can also display a play a numpy array using the `Audio` widget.

# In[ ]:


Audio(y, rate=sr)


# Let's try some experiments now. Can you guess what the following will sound like?

# In[ ]:


Audio(y, rate=sr/2)


# In[ ]:


Audio(y, rate=sr*2)


# In[ ]:


y_new, sr_new = librosa.load(example, sr=sr*2)
Audio(y_new, rate=sr_new)


# In[ ]:


y_new, sr_new = librosa.load(example, sr=sr/2)
Audio(y_new, rate=sr_new)


# ## Waveforms, amplitude vs magnitude

# A waveform is a curve showing the amplitude of the soundwave (y-axis) at time T (x-axis). Let's check out the waveform of our audio clip.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display


# In[ ]:


plt.figure(figsize=(15, 5))
librosa.display.waveplot(y, sr=sr);


# Amplitude and magnitude are often confused, but the difference is simple. Amplitude of a wave is just the distance, positive or negative, from the equilibrium (zero in our case), and magnitude is the absolute value of the amplitude. In audio we sample the amplitude.

# ## Frequency and Pitch

# Most of us remember frequency from physics as cycles per second of a wave. It's the same for sound, but really hard to see in the above image. How many cycles are there? How can there be cycles if it's not regular? The reality is that sound is extremely complex, and the above recording of human speech is the combination of many different frequencies added together. To talk about frequency and pitch, it's easier to start with a pure tone, so let's make one.
# 
# Human hearing ranges from 20hz to 20,000hz, hz=hertz=cycles per second. The higher the frequency, the more cycles per second, and the "higher" the pitch sounds to us. To demonstrate, let's make a sound at 500hz, and another at 5000hz.

# In[ ]:


import numpy as np


# In[ ]:


# Adapted from https://musicinformationretrieval.com/audio_representation.html
# An amazing open-source resource, especially if music is your sub-domain.
def make_tone(freq, clip_length=1, sr=16000):
    t = np.linspace(0, clip_length, int(clip_length*sr), endpoint=False)
    return 0.1*np.sin(2*np.pi*freq*t)
clip_500hz = make_tone(500)
clip_5000hz = make_tone(5000)


# In[ ]:


Audio(clip_500hz, rate=sr)


# In[ ]:


Audio(clip_5000hz, rate=sr)


# 500 cycles per second, 16000 samples per second, means 1 cycle = 16000/500 = 32 samples, let's see 2 cycles.

# In[ ]:


plt.figure(figsize=(15, 5))
plt.plot(clip_500hz[0:64]);


# Now let's look at 5000Hz.

# In[ ]:


plt.figure(figsize=(15, 5))
plt.plot(clip_5000hz[0:64]);


# Now let's put the two sounds together.

# In[ ]:


plt.figure(figsize=(15, 5))
plt.plot((clip_500hz + clip_5000hz)[0:64]);


# In[ ]:


Audio(clip_500hz + clip_5000hz, rate=sr)


# Pitch is a musical term that means the human perception of frequency. This concept of human perception instead of actual values seems vague and non-scientific, but it is hugely important for machine learning because most of what we're interested in, speech, sound classification, music...etc are inseparable from human hearing and how it functions.

# Let's do an experiment and increase the frequency of the above tones by 500hz each and see how much this moves our perception of them

# In[ ]:


clip_500_to_1000 = np.concatenate([make_tone(500), make_tone(1000)])
clip_5000_to_5500 = np.concatenate([make_tone(5000), make_tone(5500)])


# In[ ]:


# first half of the clip is 500hz, 2nd is 1000hz
Audio(clip_500_to_1000, rate=sr)


# In[ ]:


# first half of the clip is 5000hz, 2nd is 5500hz
Audio(clip_5000_to_5500, rate=sr)


# Notice that the pitch of the first clip seems to change more even though they were shifted by the same amount. This makes intuitive sense as the frequency of the first was doubled and the frequency of the second only increased 10%. Like other forms of human perception, hearing is not linear, it is logarithmic. This will be important later as the range of frequencies from 100-200hz convey as much information to us as the range from 10,000-20,000hz.

# ## Mel scale

# The mel scale is a human-centered metric of audio perception that was developed by asking participants to judge how far apart different tones were.
# 
# ![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/349ff3f61581b99c709f4ed29ab5e1eb6d52c98a)
# 
# | Frequency | Mel Equivalent |
# | --- | --- |
# | 20 | 0 |
# | 160 | 250 |
# | 394 | 500 |
# | 670 | 750 |
# | 1000 | 1000 |
# | 1420 | 1250 |
# | 1900 | 1500 |
# | 2450 | 1750 |
# | 3120 | 2000 |
# | 4000 | 2250 |
# | 5100 | 2500 |
# | 6600 | 2750 |
# | 9000 | 3000 |
# | 14000 | 3250 |

# ## Decibels

# Just like frequency, human perception of loudness occurs on a logarithmic scale. A constant increase in the amplitude of a wave will be perceived differently if the original sound is soft or loud.
# 
# Decibels measure the ratio of power between 2 sounds, and each 10x increase in the energy of the wave (multiplicative) results in a 10dB increase in sound (additive). Thus something that is 20dB louder has 100x (10*10) the amount of energy, something that is 25dB louder has (10^2.5) = 316.23x more energy. 
# 
# ![image.png](https://boomspeaker.com/wp-content/uploads/2018/04/DB-LEVEL-CHART-LOUD.jpg)

# ## Spectrogram - visual representation of audio
# 
# We'll plot the time on the x-axis, frequencies on the y-axis, and use the color to represent the amplitude of various frequencies.

# In[ ]:


sg0 = librosa.stft(y)
sg_mag, sg_phase = librosa.magphase(sg0)
librosa.display.specshow(sg_mag);


# Next we use the mel-scale instead of raw frequency. 

# In[ ]:


sg1 = librosa.feature.melspectrogram(S=sg_mag, sr=sr)
librosa.display.specshow(sg1);


# Next, let's use the decibel scale, and labels the x & y axes.

# In[ ]:


sg2 = librosa.amplitude_to_db(sg1, ref=np.min)
librosa.display.specshow(sg2, sr=16000, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram');


# Every point in the square represents the energy at the frequency of it's y-coordinate at the time of it's x-coordinate. 

# In[ ]:


sg2.min(), sg2.max(), sg2.mean()


# The spectrogram itself is nothing special, simply a 2d numpy array

# In[ ]:


type(sg2), sg2.shape


# In fact, we can we it as an image.

# In[ ]:


plt.imshow(sg2);


# It looks inverted because the y-axis is inverted. Also, the ticks on the y-axis now represent mel frequencies, and the ticks on the x-asis represent the actual sample length. 

# While there's a lot more to explore about audio processing, we are going to stop here, since we have successfully converted the audio into images, and now we can use same models that we use for computer vision with audio.

# Learn more about audio processing: https://www.kaggle.com/deepaksinghrawat/in-depth-introduction-to-audio-for-beginners

# In[ ]:


# Clean up
get_ipython().system('rm -rf data')

