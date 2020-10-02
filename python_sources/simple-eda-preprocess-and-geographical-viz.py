#!/usr/bin/env python
# coding: utf-8

# # Birdsong: Simple EDA 
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/Birdsong/Bewick's%20Wren%20%C2%A9%20Derek%20Hameister%20_%20Macaulay%20Library%20at%20the%20Cornell%20Lab%20of%20Ornithology%20ML214764391.png)
# 
# ---

# + <b><a href="#setup">1. Setup</a><br></b>
# + <b><a href="#basic">2. Basic EDA</a><br></b>
# + <b><a href="#p">3. Basic preprocessing</a><br></b>

# This competition is used to determine birdsong's bird of origin - it has far reaching implications in studies pertaining to wildlife and conservation. 

# <h1 id="setup">Setup</h1>

# In[ ]:


import IPython.display as ipd  # To play sound in the notebook
fname = '../input/birdsong-recognition/train_audio/aldfly/' + 'XC134874.mp3'   # Hi-hat
ipd.Audio(fname)


# In[ ]:


get_ipython().system('pip install pydub')
from pydub import AudioSegment
import wave
from scipy.io import wavfile
import pandas as pd
import re
import torch
import torchaudio
from torchaudio import transforms
import altair as alt
import plotly.express as px, plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def _generate_bar_plot_hor(df, col, title, color, w=None, h=None, lm=0, limit=100):
    cnt_srs = df[col].value_counts()[:limit]
    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], orientation = 'h',
        marker=dict(color=color))

    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    annotations = []
    annotations += [go.layout.Annotation(x=673, y=100, xref="x", yref="y", text="(Most Popular)", showarrow=False, arrowhead=7, ax=0, ay=-40)]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
plt.style.use('fivethirtyeight')
sound = AudioSegment.from_mp3(fname)
sound.export('fex1.wav', format="wav")


# Now that we have our example file in .wav format, we can proceed with our analysis:

# In[ ]:


wav = wave.open('fex1.wav')
rate, data = wavfile.read('fex1.wav')
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)


# We can take a look at this with, well, good ol' Matplotlib.

# <h1 id="basic">Basic EDA</h1>

# In[ ]:


plt.plot(data, '-', );


# We have a few jolts at the end (could be the call finally becoming louder, approacing a *crescendo*).

# In[ ]:


plt.figure(figsize=(16, 4))
plt.plot(data[:500], '.'); plt.plot(data[:500], '-');


# We see that it is fairly inconsistent in first 500 frames, with a few jolts clustered towards the end. This could potientally be a precedent for the true call.

# In[ ]:


train = pd.read_csv('../input/birdsong-recognition/train.csv')
test = pd.read_csv('../input/birdsong-recognition/test.csv')

fig = plt.figure(figsize=(16,5))
train.duration.hist(bins=100);


# Majority of files are relatively short it seems.
# 
# We can play some sample audio with IPython's capabilities to display audio in a Jupyter Notebook (very helpful). We will look at the first ten audio files:

# In[ ]:


import IPython
for i in train.filename.head(10).values:
    fname = '../input/birdsong-recognition/train_audio/aldfly/' + i
    IPython.display.display(ipd.Audio(fname))


# Hear anything? These are the various calls of the Alder Flycatcher - a bird which we are supposed to identify from its call. 

# We also have a bunch of other files to examine:

# In[ ]:


from os import *
train_dir = '../input/birdsong-recognition/train_audio/aldfly/'
train_ = [f for f in listdir(train_dir) if path.isfile(path.join(train_dir, f))]
fig = plt.figure(figsize=(15,10))
plt.suptitle('Audio examination among files')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i]
    sound = AudioSegment.from_mp3(fname)
    train_[i] = re.sub('.mp3', '', train_[i])
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(data, '-', );
    fig.add_subplot


# The files have incredibly high variance among themselves - the sound is just all over the place. This is very inconsistent and it will prove a challenge for us. We will now move to `duration`:

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,10))
ax1.hist(train.duration,color='red')
ax1.set_title('duration in train')
ax2.hist(test.seconds,color='blue')
ax2.set_title('duration in test')
plt.show()


# Even the durations are inconsistent - this will be quite a task for us. We have lengthy amounts of information in one place but virtually nothing on the other hand. We will try to strip train files of useless information. But first, since we have the goegraphical data, we will plot it on a map.

# In[ ]:


import folium, numpy as np
from folium.plugins import HeatMap,MarkerCluster
my_map_1 = folium.Map(location=[0,0], zoom_start=2, tiles='Stamen Toner')
train2 = train.head(200)

new_df = pd.DataFrame({
    'latitude': train2.dropna(axis=0, subset=['latitude','longitude'])['latitude'],
    'longitude': train2.dropna(axis=0, subset=['latitude','longitude'])['longitude']
})
hm = HeatMap(new_df,auto_play=True,max_opacity=0.8)
hm.add_to(my_map_1)
my_map_1 # display


# We can see that the study covers part of South America and most of the USA and Canada. 

# <h2 id="p">Basic preprocessing</h2>

# We will need to trim all the files to a bare minimum length of what is necessary, first we boil it down to 500 frames:

# In[ ]:


from os import *
train_dir = '../input/birdsong-recognition/train_audio/aldfly/'
train_ = [f for f in listdir(train_dir) if path.isfile(path.join(train_dir, f))]
fig = plt.figure(figsize=(15,10))
plt.suptitle('Audio examination among files')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i]
    sound = AudioSegment.from_mp3(fname)
    train_[i] = re.sub('.mp3', '', train_[i])
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(data[:500], '-', );
    fig.add_subplot


# No, no, not a good idea. What about the middle 500 frames?

# In[ ]:


fig = plt.figure(figsize=(15,10))
plt.suptitle('Audio examination among files')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i] + '.mp3'
    
    sound = AudioSegment.from_mp3(fname)
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(data[1000:1500], '-', );
    fig.add_subplot


# OK, this is better than last 500, but it is not perfect yet. 

# In[ ]:


fig = plt.figure(figsize=(15,10))
plt.suptitle('Audio examination among files')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i] + '.mp3'
    sound = AudioSegment.from_mp3(fname)
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(data[1000:2000], '-', );
    fig.add_subplot


# yes, this is even better, but what about the most inconsistent one? We can use fourier transforms on the audio data:

# In[ ]:


import warnings;warnings.filterwarnings('ignore')
from scipy.fft import fft, ifft
fig = plt.figure(figsize=(15,10))
plt.suptitle('Fourier transform among files')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i] + '.mp3'
    sound = AudioSegment.from_mp3(fname)
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(fft(data[500:1500]), '-', );
    fig.add_subplot


# In case anyone wants, here's simple yet brilliant Fourier Transform video: https://www.khanacademy.org/science/electrical-engineering/ee-signals/ee-fourier-series/v/ee-fourier-series-intro

# That is not all! We can use inverse transform as well:

# In[ ]:


import warnings;warnings.filterwarnings('ignore')
from scipy.fft import fft, ifft
fig = plt.figure(figsize=(15,10))
plt.suptitle('Fourier transform among files')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i] + '.mp3'
    sound = AudioSegment.from_mp3(fname)
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(ifft(data[500:1500]), '-', );
    fig.add_subplot


# Now we can "hose down" our data with simple signal denoising.

# In[ ]:


import pywt, numpy as np ### FROM https://www.kaggle.com/jackvial/dwt-signal-denoising
def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, sample_rate=0.02):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig
def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )
fig = plt.figure(figsize=(15,10))
plt.suptitle('Denoise')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i] + '.mp3'
    sound = AudioSegment.from_mp3(fname)
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    plt.plot(denoise_signal(data[500:1500]), '-', );
    wavfile.write(f'{i}_lin.wav', 44100,denoise_signal(data[500:1500]))
    fig.add_subplot


# One with a keen ear can notice the keen differences in the sound which our model should identify. It's your choice: if you want to use it as preprocessing or if you want to use it as data augmentation. Random noise also is possible:

# In[ ]:


fig = plt.figure(figsize=(15,10))
plt.suptitle('Denoise')
train2_ = []
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    fname = train_dir + train_[i] + '.mp3'
    sound = AudioSegment.from_mp3(fname)
    sound.export(f'{i}.wav', format="wav")
    rate, data = wavfile.read(f'{i}.wav')
    noise = np.random.normal(0, .1, data.shape)
    new = data + noise
    plt.plot(new[500:1500], '-', );
    wavfile.write(f'{i}_lin.wav', 44100,new[500:1500])
    fig.add_subplot


# In[ ]:


df = pd.DataFrame(train.ebird_code.value_counts())
df['name'] = df.index
import altair_render_script
import altair as alt
alt.Chart(df).mark_bar().encode(
    x='name',
    y='ebird_code',
    tooltip=["name","ebird_code"]
).interactive()


# A lot of birds feature 100 times in the dataset.

# In[ ]:


df = pd.DataFrame(train.author.value_counts())
df['name'] = df.index
import altair_render_script
import altair as alt
a = alt.Chart(df).mark_bar().encode(
    x='name',
    y='author',
    tooltip=["name","author"]
).interactive()
df = pd.DataFrame(train.ebird_code.value_counts())
df['name'] = df.index
import altair_render_script
import altair as alt
b = alt.Chart(df).mark_bar().encode(
    x='name',
    y='ebird_code',
    tooltip=["name","ebird_code"]
).interactive()
a & b


# We have a huge amount of authors but one man is responsible for the majority - Paul Marvin.
