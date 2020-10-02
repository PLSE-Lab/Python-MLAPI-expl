#!/usr/bin/env python
# coding: utf-8

# # Speech analysis by python
# 
# *  End-point Detection (VAE)
# * Pitch Estimation
# * Chromagram

# ## End-point Detection (VAD)
#     Goal: To detect the start and end of voice activity
#     Importance: A pre-processing step for speech-based application
#     Requirement: Low computational complexity

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Read an audio. The output **y** is samples, **fs** is sampling frequency.

# In[ ]:


wave, fs = librosa.load('../input/audio-sunday/sunday.wav', sr=None)


# In[ ]:


plt.figure(figsize=(12, 4))
librosa.display.waveplot(wave, sr=fs)
plt.show()


# In[ ]:


ipd.Audio('../input/audio-sunday/sunday.wav') # load a local WAV file


# Peform short-time fourier transform(STFT).  (Any problem with the figure below ? )

# In[ ]:


mag, phase = librosa.magphase(librosa.stft(wave, n_fft=1024, win_length=400))
plt.figure(figsize=(8,8))
librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max), x_axis='time')
plt.title('log Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()


# To calculate **RMS**.
# $$x_{rms} = \sqrt{\frac{1}{n}(x_1^2 + x_2^2 + \cdots + x_n^2)}$$
# 
# It is similar to short-time energy.
# 
# To calculate **Short-time average zero-crossing rate**.
# $$Z_n = \frac{1}{2} \sum_{m=n-N+1}^{n} \left| \text{sgn}(x[m]) -\text{ sgn}(x[ m-1 ]) \right| $$
# 
# level-crossing rate can be applied to enhance robustness.

# In[ ]:


frame_len = int(20 * fs /1000) # 20ms
frame_shift = int(10 * fs /1000) # 10ms
# calculate RMS
rms = librosa.feature.rmse(wave, frame_length=frame_len, hop_length=frame_shift)
rms = rms[0]
rms = librosa.util.normalize(rms, axis=0)

# calculate zero-crossing rate
zrc = librosa.feature.zero_crossing_rate(wave, frame_length=frame_len, hop_length=frame_shift, threshold=0)
zrc = zrc[0]
# zrc = librosa.util.normalize(zrc, axis=0)


# In[ ]:


plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
t = np.linspace(0, len(wave)/fs, len(wave))
plt.plot(t, wave, label='Waveform')
plt.legend(loc='best')

plt.subplot(3, 1, 2)
plt.plot(rms.T, label='RMS Energy')
plt.xticks([])
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.plot(zrc.T, label='zero-corssing rate')
plt.xticks([])
plt.legend(loc='best')
plt.show()


# Recall that English word 'sunday' starts with 's', a fricative. It is one of the unvoiced phonemes and has less energy than the voiced. Also it has a higher zero-crossing rate which acts as a gaussian noise. So here we can use zero-crossing rate to detect the unvoiced.
# 
# You can try different thresholds to separate speech and silence.

# In[ ]:


# set threshold of speech and silence
plt.figure(figsize=(5, 5))
n, bins, patches = plt.hist(rms.T, 20, facecolor='g', alpha=0.75)


# Based on the histogram above, we can set threshold of RMS at 0.1 and 0. 6 for zrc.

# In[ ]:


frame_idxs = np.where( (rms > 0.1) | (zrc > 0.5) )[0]


# In[ ]:


# get start-points and end-points
def getboundaries(frame_idxs):
    start_idxs = [frame_idxs[0]]
    end_idxs = []

    shapeofidxs = np.shape(frame_idxs)
    for i in range(shapeofidxs[0]-1):
        if (frame_idxs[i + 1] - frame_idxs[i]) != 1:
            end_idxs.append(frame_idxs[i])
            start_idxs.append(frame_idxs[i+1])

    end_idxs.append(frame_idxs[-1])
    # del the last boundaries if it is both start point and end point.
    if end_idxs[-1] == start_idxs[-1]:
        end_idxs.pop()
        start_idxs.pop()
    assert len(start_idxs) == len(end_idxs), 'Error! Num of start_idxs doesnt match Num of end_idxs.'
    start_idxs = np.array(start_idxs)
    end_idxs = np.array(end_idxs)
    start_t = start_idxs * frame_shift / fs
    end_t = end_idxs * frame_shift / fs
    return start_t, end_t

start_t, end_t = getboundaries(frame_idxs)


plt.figure(figsize=(12, 4))
t = np.linspace(0, len(wave)/fs, len(wave))
plt.plot(t, wave, label='Waveform')
for s, e in zip(start_t, end_t):
    plt.axvline(x=s, color='#d62728') # red vertical line
    plt.axvline(x=e, color='#2ca02c') # green vertical line
plt.legend(loc='best')
plt.show()


# However, the approach above is less robust against noise.

# In[ ]:


ipd.Audio('../input/digits/digits.wav') # load a local WAV file


# In[ ]:


wave, fs = librosa.load('../input/digits/digits.wav', sr=None)
wave = wave[:int(len(wave)/10)]
plt.figure(figsize=(12, 4))
librosa.display.waveplot(wave, sr=fs)
plt.show()


# In[ ]:


frame_len = int(20 * fs /1000) # 20ms
frame_shift = int(10 * fs /1000) # 10ms
# calculate RMS
rms = librosa.feature.rmse(wave, frame_length=frame_len, hop_length=frame_shift)
rms = rms[0]
rms = librosa.util.normalize(rms, axis=0)

# calculate zero-crossing rate
zrc = librosa.feature.zero_crossing_rate(wave, frame_length=frame_len, hop_length=frame_shift, threshold=0)
zrc = zrc[0]
# zrc = librosa.util.normalize(zrc, axis=0)

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
t = np.linspace(0, len(wave)/fs, len(wave))
plt.plot(t, wave, label='Waveform')
plt.legend(loc='best')

plt.subplot(3, 1, 2)
plt.plot(rms.T, label='RMS Energy')
plt.xticks([])
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.plot(zrc.T, label='zero-corssing rate')
plt.xticks([])
plt.legend(loc='best')
plt.show()


# BTW,  there exists a well-developed package *pyAudioAnalysis*. You can use **pip** to install it and type the following codes to do silence removal:
# 
# ```python
# # -*- coding: utf-8 -*-
# # IDE: PyCharm
# from pyAudioAnalysis import audioBasicIO as aIO
# from pyAudioAnalysis import audioSegmentation as aS
# 
# # load an audio file
# [fs, wave] = aIO.readAudioFile('./sunday.wav')
# segments = aS.silenceRemoval(wave, fs, 0.02, 0.01, smoothWindow=0.5, Weight=0.3, plot=True)
# ```
# 
# For details, pls refer to [wiki of pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/wiki/5.-Segmentation)

# ## Pitch Estimation
#         Goal: To estimate the fundamental frequency of speech or a musical note or tone.
#         Application: speaker recognition, voice coder, speech synthesis
#         Method: Autocorrelation, AMDF and so on.
#         
#         Pitch estimation is non-trivial. Here we use librosa to do pitch estimation.

# In[ ]:


frame_len = int(25 * fs /1000) # 20ms
frame_shift = int(10 * fs /1000) # 10ms
frames = librosa.util.frame(wave, frame_length=frame_len, hop_length=frame_shift)

pitches, magnitudes = librosa.core.piptrack(wave, sr=fs, hop_length=frame_shift, threshold=0.75)


# pitches[f, t] contains instantaneous frequency at freq bin f, time t.
# magnitudes[f, t] contains the corresponding magnitudes.
# 
# The output pitches can not be plotted directly as there exist several pitch candidates and we need to select the best for each frame.
# 
# To do this, we need to define two tiny functions. There is no need to understand these two functions, we can regard them as black boxes. The code comes from [github](https://github.com/tyrhus/pitch-detection-librosa-python/blob/master/script_final.py).
# 
# * function extract_max is to select the best pitch by maximizing instantaneous frequency.
# * function smooth is to low-pass filter the pitch track curve by convolution.

# In[ ]:


def extract_max(pitches, shape):
    new_pitches = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
    return new_pitches

def smooth(x,window_len=11,window='hanning'):
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]


# In[ ]:


pitch_track = extract_max(pitches, pitches.shape)
pitch_smoothtrack = smooth(pitch_track, window_len=10)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, wave)
plt.subplot(2, 1, 2)
plt.plot(pitch_smoothtrack)
plt.show()


# 
# There exist several tiny examples of pitch estimation but need some revisions to be adapted to speech. For details, please refer to https://gist.github.com/endolith/255291.

# ## Chromagram
# 
# Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.
# 
# Key points:
#     * octave equivalence
#     * 12 pitches in each octave
# 
# Simply speaking, what we need to do is to map each STFT bin to chroma (many to one).
# 
# Reference:
#     https://labrosa.ee.columbia.edu/matlab/chroma-ansyn/#1

# In[ ]:


music, fs = librosa.load('../input/piano-note/Piano.ff.C4.wav', sr=None)
chroma = librosa.feature.chroma_stft(y=music, sr=fs)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()


# In[ ]:


music, fs = librosa.load('../input/guitar/GuitarNotes.wav', sr=None)
t = np.linspace(0, len(music)/fs, num=len(music))
chroma = librosa.feature.chroma_stft(y=music, sr=fs, n_fft=2048, hop_length=160)
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, music, label='waveform')
plt.xlim([0, t[-1]])
plt.subplot(2, 1, 2)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
#plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()


# Summary:
#     In this tutorial, we show three tiny demos for speech processing by python with packages like librosa, pyAudioAnalysis, numpy. 
#     Most of them are done by simply invoking functions and it is very similar to matlab.
#     
#     
#     The advantages of python
#         * It contains more open-source packages for implementation.
#         * The source code is more readable.
#     The advantages of Matlab
#         * IDE is friendly to new-beginner.
#         * Easy to debug.
#         
#         
#  Reference: 
#  1. [A list of packages for audio/music applications](https://project-awesome.org/faroit/awesome-python-scientific-audio).
#  2. [Audio Signal Processing and Recognition](http://mirlab.org/jang/books/audioSignalProcessing/)
#  3. [VAD slide](https://mycourses.aalto.fi/pluginfile.php/146209/mod_resource/content/1/slides_07_vad.pdf)
