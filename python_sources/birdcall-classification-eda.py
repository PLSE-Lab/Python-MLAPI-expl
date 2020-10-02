#!/usr/bin/env python
# coding: utf-8

# # EDA of Meta Data

# In[ ]:


import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import plotly.express as px
import plotly.graph_objects as go
import folium
import IPython
import matplotlib.pyplot as plt


# In[ ]:


print('\n'.join(os.listdir('/kaggle/input/birdsong-recognition')))
path = '/kaggle/input/birdsong-recognition'
TRAIN_PATH = os.path.join(path, 'train_audio')
TEST_PATH = os.path.join(path, 'example_test_audio')


# In[ ]:


train_meta = pd.read_csv(os.path.join(path, 'train.csv'))
test_meta = pd.read_csv(os.path.join(path, 'test.csv'))


# ## Rating
# The rating of the audio of the quality of the audio file and the confidence of the label on the birdcall. This depends on if the researcher is able to see the bird or the amount of ambient noise present. A lower rating means a softer label.

# In[ ]:


rating_count = train_meta.groupby('rating')['xc_id'].count().reset_index()
rating_count.rename(columns={'xc_id': 'Count', 'rating': 'Rating'}, inplace=True)
fig = px.bar(rating_count, x='Rating', y='Count', title='Bar Plot of Number of Audio Recordings in Each Rating Category')
fig.show()


# ## Distribution of Labels and Training Data

# In[ ]:


labels = train_meta.groupby(['species'])['xc_id'].count().reset_index()
labels.rename(columns={'xc_id': 'Count', 'species': 'Species'}, inplace=True)
labels.sort_values(by=['Count'], inplace=True, ascending=True)
fig = px.bar(labels, x='Species', y='Count', hover_data=['Species'], color='Species')
fig.show()


# We can see that most classes have almost the exact same amount of data: 100 files. However, for a minority of classes there is data imbalance present. This could possible result in underfitting in these categories; namely, `Redhead`, `Bufflehead`, etc.

# In[ ]:


fig = px.histogram(train_meta['duration'].reset_index(), x='duration', labels={'count': 'Count', 'duration': 'Duration'},
                   title='Distribution of Duration of Audio Files')
fig.show()


# As the duration of audio file increases there is more probability that there will be ambient noise. If we train a model by splitting the audio files into 5 second partitions there is a possibility that some 5 second partitions will just be ambient noise. 

# ## Sampling Rate

# In[ ]:


train_meta.sampling_rate = train_meta.sampling_rate.apply(lambda sr: int(sr.split(' ')[0])).astype(np.uint16)
fig = px.histogram(train_meta['sampling_rate'].reset_index(), x='sampling_rate', labels={'count': 'Count', 'sampling_rate': 'Sampling Rate'},
                   title='Distribution of Sampling Rate of Audio Files')
fig.show()


# # Sample Breakdown

# In[ ]:


sample1 = train_meta.iloc[528]


# In[ ]:


IPython.display.Audio(filename=os.path.join(TRAIN_PATH, 
                                            os.path.join(sample1['ebird_code'],sample1['filename'])))


# In[ ]:


sample1


# In[ ]:


y, sr = librosa.load(os.path.join(os.path.join(TRAIN_PATH, sample1['ebird_code']), sample1['filename']), 
                     sr=sample1['sampling_rate'])
sample1_audio, _ = librosa.effects.trim(y)
time = [v/sample1['sampling_rate'] for v in range(len(sample1_audio))]
fig = px.line({'Time': time, 'Intensity': sample1_audio}, x='Time', y='Intensity', title='Wave Plot of Sample1 Audio')
fig.show()


# Let's see what the frequency domain looks like after we apply a fourier transformation. A fourier transformation takes an audio clip that is in the time domain and transforms it into the frequency domain. Here is a great video on [it](https://www.youtube.com/watch?v=spUNpyF58BY). <---

# In[ ]:


n_fft = 2048
D = np.abs(librosa.stft(sample1_audio[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fig = px.line({'Frequency': np.array(range(len(D))), 'Magnitude': D.flatten()}, x='Frequency', y='Magnitude', 
              title='Fourier Transformation of Sample1 Audio')
fig.show()


# ## Spectrogram of Sample 1

# In[ ]:


hop_length = 512
n_fft = 2048
D = np.abs(librosa.stft(sample1_audio, n_fft=n_fft,  hop_length=hop_length))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
plt.colorbar();


# It's really hard to see anything because the sounds that most humans hear are concentrated in the lower frequencies. We can adjust the intensity to decibles which is the log-scale for intensity; we can also use the [mel-scale](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) to represent frequencies. 

# In[ ]:


DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');


# This is our spectrogram ! By using spectrogram representations of our audio, we can perform image classification on these spectrograms. This makes it easy to use pre-existing DNN architectures that have been tried and proven.
