#!/usr/bin/env python
# coding: utf-8

# # Cornell Birdcall Identification
# 
# <img src="https://images.unsplash.com/photo-1512658740823-0ebb97b3b86e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80" width="700" height="300" /><br>
# 
# The goal of this notebook is to give competitors a quick overview of the Cornell Birdcall Identification challenge. After reading it you should have a good idea of the objective you are trying to solve, the data provided and the metrics you will be scored on.
# 
# Some tl;dr items to note:
# - We need to classify sounds, to understand which bird is singing.
# - The competition metric is row-wise micro averaged F1 score.
# - Train and test datasets have different structure
# 
# 
# <font size=3 color="red">Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>
# 
# To get started, here is an excellent video about bird sound classification.

# In[ ]:


from IPython.display import HTML
HTML('<iframe width="600" height="400" src="https://www.youtube.com/embed/pzmdOETnhI0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# # Acknowledgements
# 
# 1. [M5 Competition : EDA + Models](https://www.kaggle.com/tarunpaparaju/m5-competition-eda-models) ~ by Tarun Paparaju
# 2. [Cornell Birdcall: EDA Basemap Geo](https://www.kaggle.com/muhakabartay/cornell-birdcall-eda-basemap-geo) ~ by Mukharbek Organokov

# # Contents
# 
# * [<font size=4>The dataset</font>](#1)
# 
# * [<font size=4>EDA</font>](#2)
#     * [Make imports](#2.1)
#     * [Load the data](#2.2)
#     * [Listen to the birds](#2.3)
#     * [Sample durations histogram](#2.4)
#     * [Where do the birds come from?](#2.5)
#     * [Some wrangling on the bird songs data](#2.6)
#     * [Denoising bird sounds](#2.7)
# 
# * [<font size=4>Baseline submission</font>](#3)
# 
# * [<font size=4>Takeaways</font>](#4)
# 
# * [<font size=4>Relevant Competitions Collection</font>](#5)
# 
# * [<font size=4>Ending Note</font>](#6)

# # The dataset <a id="1"></a>
# 
# The dataset consists of .csv files and some sound recordings in the folder structure.
# 
# * <code>example_test_audio/</code> - Contains examples of how the recordings look like on the notebook inference
# 
# * <code>train_audio/</code> - Folder structure containing multiple recordings for many bird species
# 
# * <code>train.csv</code> - Contains metadata for the train set
# 
# * <code>example_test_audio_summary.csv, example_test_audio_metadata.csv, test.csv</code> - Examples of the files which will be provided to the notebook on the hidden test set run.
# 
# Once again, in this competition we are trying to classify bird sounds on multiple classes (bird species)

# ## EDA

# ### Make imports <a id="2.1">

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import librosa

import pywt
from statsmodels.robust import mad

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


PATH = '../input/birdsong-recognition/'
os.listdir(PATH)


# ### Load the data <a id="2.2">
# 
# I will firstly print all the tables, so we can understand the data sources

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv(os.path.join(PATH, 'train.csv'))\ntest = pd.read_csv(os.path.join(PATH, 'test.csv'))\nsubmission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))\nexample_test_audio_summary = pd.read_csv(os.path.join(PATH, 'example_test_audio_summary.csv'))\nexample_test_audio_metadata = pd.read_csv(os.path.join(PATH, 'example_test_audio_metadata.csv'))")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


example_test_audio_summary.head()


# In[ ]:


example_test_audio_metadata.head()


# In[ ]:


submission.head()


# ### Listen to the birds <a id="2.2">
# As can be seen below, there are 264 bird species in total. I will make an audio samples of birds' singing, to get the idea of how the training data looks like.

# In[ ]:


birds = np.array(os.listdir(os.path.join(PATH, 'train_audio')))
print(f'All bird species {len(birds)}:\n')
print(*birds, sep = '\n')


# In[ ]:


import IPython.display as ipd  
sample_birds = birds[np.random.randint(0, len(birds), 3)]
songs = []
for bird in sample_birds:
    bird_path = np.array(os.listdir(os.path.join(os.path.join(PATH, 'train_audio', bird))))
    bird_song = bird_path[np.random.randint(0, len(bird_path), 1)][0]
    songs.append(os.path.join(os.path.join(PATH, 'train_audio', bird, bird_song)))


# In[ ]:


print(f'{sample_birds[0]} is singing')
ipd.Audio(songs[0])


# In[ ]:


print(f'{sample_birds[1]} is singing')
ipd.Audio(songs[1])


# In[ ]:


print(f'{sample_birds[2]} is singing')
ipd.Audio(songs[2])


# ### Sample durations histogram <a id="2.3">

# In[ ]:


fig = px.histogram(x=train['duration'].values, title = 'Duration of bird songs in seconds')
fig.show()


# ### Where do the birds come from? <a id="2.4">
# As you can see, most of the bird songs originate from the United States

# In[ ]:


fig = px.histogram(x=train['country'].values, title = 'Countries of the birds')
fig.show()


# ### When were these recordings made? <a id="2.5">

# In[ ]:


# you can notice a way to fill NaNs, as well as some really strange dates
df = train.groupby('date').apply(len)
print(df)
fig = px.line(x=df.index[4:], y=df.values[4:], title='Bird song recordings wrt date')
fig.show()


# ### Some wrangling on the bird songs data <a id="2.6">

# In[ ]:


from warnings import filterwarnings; filterwarnings('ignore')
waves = []
for i in range(3):
    y, sr = librosa.load(songs[i])
    wave = y[::100]
    waves.append(y)
    fig = px.line(x=np.arange(len(wave)), y=wave, title=f'{sample_birds[i]} song wave')
    fig.show()


# ### Denoising bird sounds <a id="2.7">
# * Credits go to [this excellent kernel](https://www.kaggle.com/tarunpaparaju/m5-competition-eda-models)

# #### Wavelet transform

# In[ ]:


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')


# In[ ]:


x_1, x_2, x_3 = waves[0][::100], waves[1][::100], waves[2][::100]

y_w1 = denoise_signal(x_1)
y_w2 = denoise_signal(x_2)
y_w3 = denoise_signal(x_3)


fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="mediumaquamarine"), showlegend=False,
               name="Original signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), y=y_w1, mode='lines', marker=dict(color="darkgreen"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), y=y_w2, mode='lines', marker=dict(color="purple"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="lightskyblue"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), y=y_w3, mode='lines', marker=dict(color="navy"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) bird sounds")
fig.show()


# #### Average smoothing

# In[ ]:


def average_smoothing(signal, kernel_size=3, stride=1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
    return np.array(sample)


# In[ ]:


y_w1 = average_smoothing(x_1)
y_w2 = average_smoothing(x_2)
y_w3 = average_smoothing(x_3)


fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="mediumaquamarine"), showlegend=False,
               name="Original signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), y=y_w1, mode='lines', marker=dict(color="darkgreen"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), y=y_w2, mode='lines', marker=dict(color="purple"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="lightskyblue"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), y=y_w3, mode='lines', marker=dict(color="navy"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) bird sounds")
fig.show()


# ## Baseline model <a id="3">

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import librosa

import pywt
from statsmodels.robust import mad
from warnings import filterwarnings; filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.io import wavfile
import subprocess

from tqdm.notebook import tqdm
import librosa

PATH = '../input/birdsong-recognition/'
TEST_FOLDER = '../input/birdsong-recognition/test_audio/'
os.listdir(PATH)
RANDOM_SEED = 4444


# In[ ]:


def load_test_clip(path, start_time, duration=5):
    return librosa.load(path, offset=start_time, duration=duration)[0]

def make_prediction(y, le, model):
    feats = np.array([np.min(y), np.max(y), np.mean(y), np.std(y)]).reshape(1, -1)
    return le.inverse_transform(model.predict(feats))[0]


# In[ ]:


train = pd.DataFrame(columns = ['min', 'max', 'mean', 'std', 'target'])
nRows = 0
for bird in tqdm(os.listdir(os.path.join(PATH, 'train_audio'))):
    for audio in  os.listdir(os.path.join(PATH, 'train_audio', bird)):
        path = os.path.join(PATH, 'train_audio', bird, audio)
        subprocess.call(['ffmpeg', '-y', '-i', f'{path}', f'/kaggle/working/temp.wav'])
        _, y = wavfile.read('/kaggle/working/temp.wav')
        train.loc[nRows] = [np.min(y), np.max(y), np.mean(y), np.std(y), bird]
        nRows += 1


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['target'] = le.fit_transform(train['target'].values)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth = 6, random_state = RANDOM_SEED)
model.fit(train.drop(columns = 'target'), train['target'].values)


# In[ ]:


try:
    preds = []
    for index, row in test.iterrows():
        # Get test row information
        site = row['site']
        start_time = row['seconds'] - 5
        row_id = row['row_id']
        audio_id = row['audio_id']

        # Get the test sound clip
        if site == 'site_1' or site == 'site_2':
            y = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time)
        else:
            y = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)

        # Make the prediction
        pred = make_prediction(y, le, model)

        # Store prediction
        preds.append([row_id, pred])
except:
     preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
preds = pd.DataFrame(preds, columns=['row_id', 'birds'])


# In[ ]:


preds.to_csv('submission.csv', index=False)


# ## Takeaways <a id="4">
# 
# * Train and test have different structure
# * Most of the bird songs originate from the United States
# * There is a really small portion of strange dates in the data
# * From the sample bird songs waves, it seems that they can be pretty well separated

# ## Relevant Competitions Collection  <a id="5">
#     
# * [Don't call me turkey](https://www.kaggle.com/c/dont-call-me-turkey)
# * [Freesound audio tagging](https://www.kaggle.com/c/freesound-audio-tagging-2019)
# * [Multi-label Bird Species Classification](https://www.kaggle.com/c/multilabel-bird-species-classification-nips2013)
# 
#  

# ## Ending note <a id = "6">
#   
# <font color="red" size=4>This concludes my kernel. This motivates a lot :)</font>

# In[ ]:




