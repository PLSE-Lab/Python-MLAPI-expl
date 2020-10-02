#!/usr/bin/env python
# coding: utf-8

# 
# # <font size='7' color='blue'>Cornell BirdSong Recognition</font>

# <img src="https://lh3.googleusercontent.com/proxy/UWuIXQz5U-mEDO1x18b2adWKk3KG9hDFNpXx9_GS-RWM7dSl_W4MaYKh1ZbnuPfM3_P-gllYn7G46N1kmzYk3qs5zTNVSvIluXsvigFEMika_Djh2jmYMuBwbZSx5Z2JheFgLNGeFW2zskaJiVdRhdBJMXRHuPZq3rpXspsGH8NwiCT_1ndBLQ9EXWLAj6H-JUQNPUKbdp4eloFyKLNeQAftOJm0M12FUkSAv8dgcZFLRiGVwxvTkDGgp9JLP8QjNlBgtZ9oeFWUGLLoS_mwen1EUyG7llE0ZssX" alt="Meatball Sub" width="500"/>

# ## <font size='7' color='red'>Contents</font> 
# 
# * [Basic Exploratory Data analysis](#1)  
# 
#     * [Getting started - import data]()
#     * [Missing values]()
#     * [Rate of each specie]()
#     * [where the recordings come from, the heat map]()
#     * [Dates on which samples are collected]()
#     * [Length/duration of record]()
#     * [Author distribution]()
#  
#  
# * [Audio Data analysis](#2) 
# 
#      * [Playing audio]()
#      * [Visualizing audio in 2D]()
#      * [Spectrogram analysis]()
#      * [MelSpectrogram analysis]()
#  

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from math import *

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from ipywidgets import widgets
from ipywidgets import *

init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[ ]:


train = pd.read_csv("../input/birdsong-recognition/train.csv",delimiter=",",encoding="latin", engine='python')
test = pd.read_csv("../input/birdsong-recognition/test.csv",delimiter=",",encoding="latin", engine='python')
audio_summary = pd.read_csv("../input/birdsong-recognition/example_test_audio_summary.csv",delimiter=",",encoding="latin", engine='python')
audio_metadata = pd.read_csv("../input/birdsong-recognition/example_test_audio_metadata.csv",delimiter=",",encoding="latin", engine='python')


# In[ ]:


train.head(5)


# ### Missing values 

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.isna(), cbar=False)


# In[ ]:


missing_rate_train = (train.isna().sum()/train.shape[0]).sort_values()
nb_missing = train.isna().sum().sort_values()
print(f'{"Variable" :-<25} {"missing_rate_train":-<25} {"Number of missing values":-<25}')
for n in range(len(missing_rate_train)):
    print(f'{missing_rate_train.index[n] :-<25} {missing_rate_train[n]:-<25} {nb_missing[n]:-<25}')


# In[ ]:


train.columns


# In[ ]:


len(train["species"].unique())


# <div class="alert alert-block alert-warning">  
# <b> Observation 1 :</b> We have 264 species !! 
# </div>

# ### Rate of each specie

# In[ ]:


rate = train["species"].value_counts().sort_values()/264
print(f'{"Target" :-<40} {"rate":-<20}')
for n in range(len(rate)):
    print(f'{rate.index[n] :-<40} {rate[n]}')


# In[ ]:


train["species"].value_counts().sort_values().iplot(kind="bar",)


# ### Create a heat map to present of records

# In[ ]:


longitude = pd.to_numeric(train['longitude'], errors='coerce')
latitude = pd.to_numeric(train['latitude'], errors='coerce')
df = pd.concat([longitude,latitude],axis=1)


# In[ ]:


import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
f = folium.Figure(width=1000, height=500)

longitude = pd.to_numeric(train['longitude'], errors='coerce')
latitude = pd.to_numeric(train['latitude'], errors='coerce')
df = pd.concat([longitude,latitude],axis=1).dropna()
m = folium.Map(location=[40, 0], zoom_start=2).add_to(f)
# Add a heatmap to the base map
HeatMap(data=df[['latitude', 'longitude']], radius=10).add_to(m)
m


# ### Playback used

# In[ ]:


train['playback_used'].fillna('Missing',inplace=True)
labels=train['playback_used'].value_counts().index
values=train['playback_used'].value_counts().values
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial'
                            )])
fig.show()


# ### Date of records - Dates on which samples are collected

# In[ ]:


train['date'] = train['date'].apply(pd.to_datetime,format='%Y-%m-%d', errors='coerce')
train['date'].value_counts().plot(figsize=(25, 6))


# In[ ]:


plt.figure(figsize=(25, 6))
ax = sns.countplot(train['date'].dt.year.dropna().apply(lambda x : int(x)), palette="hls")


# ### Length of record

# In[ ]:


train["length"].value_counts().sort_values().iplot(kind="bar",)


# ### Author distribution

# In[ ]:


train["author"].value_counts().sort_values().iplot(kind="bar",)


# ### Ebird code distribution

# In[ ]:


train["ebird_code"].value_counts().sort_values().iplot(kind="bar")


# ## Audio information / exploration 
# One of the most popular packages in Python to do music analysis is called librosa. I am inviting you to watch this video on youtube.

# In[ ]:


from IPython.display import YouTubeVideo

YouTubeVideo('MhOdbtPhbLU', width=800, height=300)


# ### Loading audio file:

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import librosa
import librosa.display
audio_data = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'
x , sr = librosa.load(audio_data)
print(x.shape, sr)


# In[ ]:


librosa.load(audio_data,sr)


# ### Playing Audio

# In[ ]:


import IPython.display as ipd
ipd.Audio(audio_data)


# In[ ]:


from random import sample 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
colorsName = [name for hsv, name in by_hsv]


# In[ ]:


class AudioProcessing:
    
    def ReadAudio(self,ebird_code,filename):
        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename
        x , sr = librosa.load(audio_file)
        return x,sr
    
    def LoadAudio(self,audio_file,sr):
        return librosa.load(audio_file,sr)
        
    def PlayingAudio(self,ebird_code,filename):
        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename
        x , sr = librosa.load(audio_file)
        librosa.load(audio_file,sr)
        return ipd.Audio(audio_file)
                         
    def DisplayWave(self,ebird_code,filename):
        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename
        y, sr = librosa.load(audio_file)
        whale_song, _ = librosa.effects.trim(y)
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(whale_song, sr=sr)
        plt.show()
                         
    def DisplaySpectogram(self,ebird_code,filename):
        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename
        x , sr = librosa.load(audio_file)
        Xdb = librosa.amplitude_to_db(abs(librosa.stft(x)))
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()               
    def PlotSampleWave(self,nrows,captions,df):
        ncols=1
        f, ax = plt.subplots(nrows,ncols=ncols,figsize=(ncols*12,nrows*3))
        i = 0
        colors = sample(colorsName,nrows)
        for c in captions:
            samples = df[df['ebird_code']==c]['filename'].sample(ncols).values
            audio_file = "../input/birdsong-recognition/train_audio" + "/" + c + "/" + samples[0]
            y, sr = librosa.load(audio_file)
            whale_song, _ = librosa.effects.trim(y)
            librosa.display.waveplot(whale_song, sr=sr, color = colors[i],ax=ax[i])
            i = i + 1
        for i, name in zip(range(nrows), captions):
            ax[i].set_ylabel(name, fontsize=15)
        plt.tight_layout()
        plt.show()
    
    def PlotSampleSpectrogram(self,nrows,captions,df):
        ncols=1
        f, ax = plt.subplots(nrows,ncols=ncols,figsize=(ncols*12,nrows*3))
        i = 0
        colors = sample(colorsName,nrows)
        for c in captions:
            samples = df[df['ebird_code']==c]['filename'].sample(ncols).values
            audio_file = "../input/birdsong-recognition/train_audio" + "/" + c + "/" + samples[0]
            x, sr = librosa.load(audio_file)
            X = librosa.stft(x)
            Xdb = librosa.amplitude_to_db(abs(X))
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz',ax=ax[i])
            i = i + 1
        for i, name in zip(range(nrows), captions):
            ax[i].set_ylabel(name, fontsize=15)
        plt.tight_layout()
        plt.show()
        
    def PlotSampleMelSpectrogram(self,nrows,captions,df):
        ncols=1
        f, ax = plt.subplots(nrows,ncols=ncols,figsize=(ncols*12,nrows*3))
        i = 0
        colors = sample(colorsName,nrows)
        for c in captions:
            samples = df[df['ebird_code']==c]['filename'].sample(ncols).values
            audio_file = "../input/birdsong-recognition/train_audio" + "/" + c + "/" + samples[0]
            x, sr = librosa.load(audio_file)
            S = librosa.feature.melspectrogram(x, sr)
            log_S = librosa.power_to_db(S, ref=np.max)

            librosa.display.specshow(log_S, sr = sr, hop_length = 500, x_axis = 'time', y_axis = 'log', cmap = 'rainbow', ax=ax[i])
            i = i + 1
        for i, name in zip(range(nrows), captions):
            ax[i].set_ylabel(name, fontsize=15)
        plt.tight_layout()
        plt.show()
            


# ### 1 - Plot a simple of sound waves 

# Comparing wave curve for different birds :

# In[ ]:


N = 5
ebird_code_simple = sample(list(train["ebird_code"].unique()),N)
AudioProcessing().PlotSampleWave(nrows=N,captions=ebird_code_simple,df=train)


# ### 2 - Spectrogram 

# **According to wikipedia :** 
# 
# - A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams. When the data is represented in a 3D plot they may be called waterfalls.
# 
# - Spectrograms are used extensively in the fields of music, linguistics, sonar, radar, speech processing, seismology, and others. Spectrograms of audio can be used to identify spoken words phonetically, and to analyse the various calls of animals.
# 
# - A spectrogram can be generated by an optical spectrometer, a bank of band-pass filters, by Fourier transform or by a wavelet transform (in which case it is also known as a scaleogram or scalogram).
# 
# - A spectrogram is usually depicted as a heat map, i.e., as an image with the intensity shown by varying the colour or brightness.

# Comparing spectrogram curve for different birds :

# In[ ]:


AudioProcessing().PlotSampleSpectrogram(nrows=N,captions=ebird_code_simple,df=train)


# ### 3 - MelSpectrogram 

# **According to wikipedia :**
# 
# - The mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
# 
# - Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). The difference between the cepstrum and the mel-frequency cepstrum is that in the MFC, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal cepstrum. This frequency warping can allow for better representation of sound, for example, in audio compression.
# 
# - MFCCs are commonly derived as follows:
#     1. Take the Fourier transform of (a windowed excerpt of) a signal.
#     2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
#     3. Take the logs of the powers at each of the mel frequencies.
#     4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
#     5. The MFCCs are the amplitudes of the resulting spectrum.

# In[ ]:


AudioProcessing().PlotSampleMelSpectrogram(nrows=N,captions=ebird_code_simple,df=train)


# ## To be continued ... 
