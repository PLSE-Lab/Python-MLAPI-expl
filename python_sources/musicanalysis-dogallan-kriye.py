#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import
import librosa
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
#Try to indentify its genre,tempo,instruments,mood,time signature,key signature,chord progression,tuning frequency,song structure


# sampling of tasks found in music information retrieval:
# - fingerprinting
# - cover song detection
# - genre recognition
# - transcription
# - recommendation
# - symbolic melodic similarity
# - mood
# - source separation
# - instrument recognition
# - pitch tracking
# - tempo estimation
# - score alignment
# - song structure/form
# - beat tracking
# - key detection
# - query by humming

# In[ ]:


cd ../input


# In[ ]:


ipd.YouTubeVideo('SEyzZw7xiPo')


# In[ ]:


garry, sr = librosa.load('LETS TALK (DO GALLAN ) _ Full Video _ GARRY SANDHU _ FRESH MEDIA RECORDS.mp3')


# In[ ]:


ipd.Audio('LETS TALK (DO GALLAN ) _ Full Video _ GARRY SANDHU _ FRESH MEDIA RECORDS.mp3') # load a local WAV file


# In[ ]:


librosa.get_duration(garry)


# In[ ]:


yt, index = librosa.effects.trim(garry)
print(librosa.get_duration(garry), librosa.get_duration(yt))


# In[ ]:


ipd.Audio('LETS TALK (DO GALLAN ) _ Full Video _ GARRY SANDHU _ FRESH MEDIA RECORDS.mp3') # load a local WAV file


# In[ ]:


garry=yt
garry


# In[ ]:


garry.shape
sr


# In[ ]:


print(yt.shape,sr)#total length/values in the array# sample rate of this music


# In[ ]:


plt.figure(figsize=(25, 10))
librosa.display.waveplot(yt, sr=sr)


# In[ ]:


n0 = 6500
n1 = 7000
plt.figure(figsize=(25, 10))
plt.plot(garry[n0:n1])


# In[ ]:


Garry = librosa.stft(garry)#compute stft
Xdb = librosa.amplitude_to_db(abs(Garry))#computing log amplitude as the lower graph look srather boring
plt.figure(figsize=(25, 10))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#full spectrogram


# In[ ]:


garry.shape


# In[ ]:


#let's look more closely
Garry = librosa.stft(garry[20000:21000])#compute stft
Xdb = librosa.amplitude_to_db(abs(Garry))#computing log amplitude as the lower graph look srather boring
plt.figure(figsize=(25, 10))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#so it's a lot of bar graphs all in one


# In[ ]:


#Let's try to detect onsets now
onset_frames = librosa.onset.onset_detect(garry, sr=sr)
print(onset_frames)


# In[ ]:


onset_times = librosa.frames_to_time(onset_frames, sr=sr)
print(onset_times)


# In[ ]:


onset_samples = librosa.frames_to_samples(onset_frames)
print(onset_samples)


# In[ ]:


# Use the `length` parameter so the click track is the same length as the original signal
clicks = librosa.clicks(times=onset_times, length=len(garry))


# In[ ]:


# Play the click track "added to" the original signal
ipd.Audio(garry+clicks, rate=sr)#other than for just feel using onset detection here is kind of mistake;)


# In[ ]:


#Segmentation show and then creating segments of 50 segments at beginning of each onset
#concatanete also add 100 ms second into each segments then we concatenate them all into one signal
frame_sz = int(0.1*sr)
segments = numpy.array([garry[i:i+frame_sz] for i in onset_samples])
def concatenate_segments(segments, sr=22050, pad_time=0.300):
    padded_segments = [numpy.concatenate([segment, numpy.zeros(int(pad_time*sr))]) for segment in segments]
    return numpy.concatenate(padded_segments)
concatenated_signal = concatenate_segments(segments, sr)
ipd.Audio(concatenated_signal, rate=sr)


# In[ ]:


#let's calculate zero crossing rate for that segments
#zero crossing rate indicates the number of times that signal crosses the horizontal axis
zcrs = [sum(librosa.core.zero_crossings(segment)) for segment in segments]
print(zcrs)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.plot(zcrs)


# In[ ]:


ind = numpy.argsort(zcrs)
print(ind)


# In[ ]:


concatenated_signal = concatenate_segments(segments[ind], sr)


# In[ ]:


ipd.Audio(concatenated_signal, rate=sr)


# In[ ]:


#finding MFCC
garry = librosa.feature.mfcc(yt, sr=sr)
print(garry.shape)
#in this case,mff computed 20 MFCCS over 11343 frames


# The very first MFCC, the 0th coefficient, does not convey information relevant to the overall shape of the spectrum. It only conveys a constant offset, i.e. adding a constant value to the entire spectrum. Therefore, many practitioners will discard the first MFCC when performing classification. For now, we will use the MFCCs as is.

# In[ ]:


plt.figure((figsize=(25,, 10)))
librosa.display.specshow(garry, sr=sr, x_axis='time')


# In[ ]:


import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display
import essentia, essentia.standard as ess


# In[ ]:


garry = sklearn.preprocessing.scale(garry, axis=1)
print(garry.mean(axis=1))
print(garry.var(axis=1))


# In[ ]:


plt.figure(figsize=(25, 10))
librosa.display.specshow(garry, sr=sr, x_axis='time')


# In[ ]:




