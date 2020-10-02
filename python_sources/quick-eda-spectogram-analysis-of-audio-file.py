#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
import scipy


from IPython.display import Audio
from IPython.core.display import HTML
from pandas_profiling import ProfileReport
from scipy.io import wavfile


# ## Quick peek at the markdown

# In[ ]:


# no of bird classes
get_ipython().system(' ls /kaggle/input/birdsong-recognition/train_audio/ | wc -l')


# In[ ]:


train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


train_summary = pd.read_csv('/kaggle/input/birdsong-recognition/example_test_audio_summary.csv')
train_summary.head()


# ## Quick EDA
# 
# using pandas profile report

# In[ ]:


train_profile = ProfileReport(train)


# In[ ]:


train_profile.to_widgets()


# In[ ]:


# Press the output to see the complete report
train_profile


# ## Analysing the sound with EDA

# In[ ]:


Audio('/kaggle/input/birdsong-recognition/train_audio/eawpew/XC148566.mp3')


# - loading audio with librosa
# - finding beat_time

# In[ ]:


y, sr = librosa.load('/kaggle/input/birdsong-recognition/train_audio/eawpew/XC148566.mp3')
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

beat_times = librosa.frames_to_time(beat_frames, sr=sr)


# - find sample_rate
# - find signal_length
# - duration

# In[ ]:


print(f"Sample rate  :", sr)
print(f"Signal Length:{len(y)}")
print(f"Duration     : {len(y)/sr}seconds")


# In[ ]:


plt.figure(figsize=(15, 5))
librosa.display.waveplot(y, sr=sr)


# ## Spectogram
# 
# - checking what is characteristics of frequency

# In[ ]:


sg0 = librosa.stft(y)
sg_mag, sg_phase = librosa.magphase(sg0)
display(librosa.display.specshow(sg_mag))


# In[ ]:


sg1 = librosa.feature.melspectrogram(S=sg_mag, sr=sr)
display(librosa.display.specshow(sg1))


# In[ ]:


sg2 = librosa.amplitude_to_db(sg1, ref=np.min)
librosa.display.specshow(sg2)


# ## What's inside a spectogram

# In[ ]:


# code adapted from the librosa.feature.melspectrogram documentation
librosa.display.specshow(sg2, sr=16000, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')


# In[ ]:


sg2.min(), sg2.max(), sg2.mean()


# ## Fourier Transformation

# In[ ]:



# Code adapted from https://musicinformationretrieval.com/fourier_transform.html and the original
# implementation of fastai audio by John Hartquist at https://github.com/sevenfx/fastai_audio/
def fft_and_display(signal, sr):
    ft = scipy.fftpack.fft(signal, n=len(signal))
    ft = ft[:len(signal)//2+1]
    ft_mag = np.absolute(ft)
    f = np.linspace(0, sr/2, len(ft_mag)) # frequency variable
    plt.figure(figsize=(13, 5))
    plt.plot(f, ft_mag) # magnitude spectrum
    plt.xlabel('Frequency (Hz)')


# In[ ]:


fft_and_display(y, sr)


# ## Making a submission

# In[ ]:


sub = pd.read_csv('/kaggle/input/birdsong-recognition/sample_submission.csv')
sub.to_csv('submission.csv', index=False)

