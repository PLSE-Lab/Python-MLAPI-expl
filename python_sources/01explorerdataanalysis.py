#!/usr/bin/env python
# coding: utf-8

# ### Intro
# The data set for audio tagging 2019 is designed with label noise. This notebook is to explore the basic information, the audio features of three subsets.
# 
# Thanks to below kagglers, some functions are re-orgnized here. Wish this notebook could be helpful.
# 
# ref:
# * https://www.kaggle.com/maxwell110/explore-multi-labeled-data
# * https://www.kaggle.com/dude431/beginner-s-visualization-and-removing-uniformative

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import IPython.display as ipd
import librosa


# ### 1. Basic information
# Basice information about the data set, such as head of data, numbers, label distributions and so on.

# In[ ]:


# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# !ls ../input/


# In[ ]:


train = pd.read_csv("../input/train_curated.csv")
train['is_curated'] = True
train_noisy = pd.read_csv("../input/train_noisy.csv")
train_noisy['is_curated'] = False
train = pd.concat([train, train_noisy], axis=0)
del train_noisy
# print(train.head())
# print(train.tail())


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train['n_label'] = train.labels.str.split(',').apply(lambda x: len(x))
print(train.head(10))
print(train.tail(6))
print("Number of train examples: ", train.shape[0])
print("In curated subset: ", train[train.is_curated == True].shape[0])
print("In noisy subset: ", train[train.is_curated == False].shape[0])
# print("Number of classes:", len(set(train.labels)))


# In[ ]:


test = pd.read_csv("../input/sample_submission.csv")
# print(test.head())
test.head()


# In[ ]:


print("Number of test examples: ", test.shape[0],
      "\nNumber of classes: ", len(set(test.columns[1:])))
print(set(test.columns[1:]))


# ### Audio duration
# 
# It's introduced that durations of samples in curated subset are from 0.3s to 30s, while those in noisy subset are from 1s to 15s, with the vast majority lasting 15s.

# In[ ]:


import wave
SAMPLE_RATE = 44100

train_1 = train[train.is_curated == True].sort_values('labels').reset_index()
train_1['nframes'] = train_1['fname'].apply(lambda f: 
    wave.open('../input/train_curated/' + f).getnframes()/SAMPLE_RATE)
train_1.head()


# In[ ]:


train_2 = train[train.is_curated == False].sort_values('labels').reset_index()
train_2['nframes'] = train_2['fname'].apply(lambda f: 
    wave.open('../input/train_noisy/' + f).getnframes()/SAMPLE_RATE)
train_2.head()


# In[ ]:


test_new = test
test_new['nframes'] = test_new['fname'].apply(lambda f: 
    wave.open('../input/test/' + f).getnframes()/SAMPLE_RATE)
test_new.head()


# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(16,5))
train_1.nframes.hist(bins=100, grid=True, rwidth=0.5, color='blue', ax=axes[0])
train_2.nframes.hist(bins=100, grid=True, rwidth=0.5, color='black', ax=axes[1])
test_new.nframes.hist(bins=100, grid=True, rwidth=0.5, color='red', ax=axes[2])
plt.suptitle('Duration Distribution in curated, noisy and test set', ha='center', fontsize='large');


# Majority of the audio files are short than 10s, when we crop audio as cnn train data, 4-8 seconds should be OK.
# 
# There are an abnormal length in the train histogram.

# In[ ]:


train_1.query("nframes > 30")


# In[ ]:


ipd.Audio( '../input/train_curated/' + '77b925c2.wav')


# In[ ]:


del train_1
del train_2
del test_new


# ### Train set labels distributions:
# Samples in curated subset have 1, 2, 3, 4, 6 labels, while samples in noisy subset have 1, 2, 3, 4, 5, 6, 7 labels. Most samples have a single label.

# In[ ]:


train.query('is_curated == True').n_label.value_counts()


# In[ ]:


train.query('is_curated == False').n_label.value_counts()


# In[ ]:


cat_gp = train[train.n_label == 1].groupby(
    ['labels', 'is_curated']).agg({'fname':'count'}).reset_index()
cat_gpp = cat_gp.pivot(index='labels', columns='is_curated', values='fname').reset_index().set_index('labels')

plot = cat_gpp.plot(
    kind='barh',
    title="Number of samples per category",
    stacked=True,
    color=['deeppink', 'darkslateblue'],
    figsize=(15,20))
plot.set_xlabel("Number of Samples", fontsize=20)
plot.set_ylabel("Label", fontsize=20);


# #### Wordcloud for labels

# In[ ]:


from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(train.labels))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Labels", fontsize=35)
plt.axis("off")
plt.show() 


# ### Audio features
# Try to listen samples of specific label, check waveform and spectrogram together.

# In[ ]:


import librosa, librosa.display
import matplotlib.pyplot as plt
import os
import IPython
SAMPLE_RATE = 44100

def load_and_show(path, fname):
    plt.figure(figsize=(10,3))
    wav, sr = librosa.core.load(os.path.join(path, fname))
#     melspec = librosa.feature.melspectrogram(
#         librosa.resample(wav, sr, SAMPLE_RATE),
#         sr=SAMPLE_RATE/2, n_fft = 1024,
#         hop_length=512, n_mels= 128
#     )
#     logmel = librosa.core.power_to_db(melspec)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
    # https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/91827#latest-529419
#     D = librosa.pcen(np.abs(librosa.stft(wav)))
    # CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
    plt.subplot(1,2,1)
    librosa.display.waveplot(wav, sr=SAMPLE_RATE)
    plt.subplot(1,2,2)
    librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='linear')
    plt.title(os.path.join(path, fname))


# In[ ]:


# sampling an audio in train_curated
# choose a label randomly
samp = train[(train.n_label == 1) & (train.is_curated == True)].sample(1)
specific_label = samp.labels.values[0]

# for the specific label, choose several samples
samples = train[(train.n_label == 1) 
               & (train.is_curated == True) 
               & (train.labels == specific_label)].sample(5)

# Listen and check the wave, spectrogram of the samples.
print('Label:', specific_label)
for fname in list(samples.fname):
#     print('../input/train_curated/{}'.format(fname))
    load_and_show('../input/train_curated', fname)
    IPython.display.display(ipd.Audio('../input/train_curated/{}'.format(fname)))


# In[ ]:


# sampling an audio in train_noisy
# for the specific label, choose several samples
samples = train[(train.n_label == 1) 
               & (train.is_curated == False) 
               & (train.labels == specific_label)].sample(5)

# Listen and check the wave, spectrogram of the samples.
print('Label:', specific_label)
for fname in list(samples.fname):
    print('../input/train_noisy/{}'.format(fname))
    load_and_show('../input/train_noisy', fname)
    IPython.display.display(ipd.Audio('../input/train_noisy/{}'.format(fname)))


# We can try some different labels, it seems a lot of noisy data have wrong labels.
# 
# To let the noisy data make positive effect, designed loss function or semi-supervised learning should be used.

# ### Multi labels
# 
# About 1/5 of samples in curated and noisy subset are with multi labels, so let's check the samples with multi labels.

# In[ ]:


cat_gp = train[(train.n_label > 1) & (train.is_curated == True)].groupby('labels').agg({'fname':'count'})
cat_gp.columns = ['counts']

plot = cat_gp.sort_values(ascending=True, by='counts').plot(
    kind='barh',
    title="Number of Audio Samples per Category",
    color='deeppink',
    figsize=(15,30))
plot.set_xlabel("Number of Samples", fontsize=20)
plot.set_ylabel("Label", fontsize=20);


# There are only about 10 kinds of multi labels whose samples are more than 10.

# #### 2 label conditions

# In[ ]:


label_set = set(train.loc[(train.n_label == 2) & (train.is_curated == True), 'labels']) & set(
    train.loc[(train.n_label == 2) & (train.is_curated == False), 'labels'])

label_samp = np.random.choice(list(label_set), 1)[0]
samp = train[(train.labels == label_samp) & (train.is_curated == True)].sample(1)
print(label_samp)
IPython.display.display(ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0])))
load_and_show('../input/train_curated', samp.fname.values[0])


# In[ ]:


# sampling an audio in train_noisy
samp_n = train[(train.labels == label_samp) & (train.is_curated == False)].sample(1)
print(samp_n.labels.values[0])
IPython.display.display(ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0])))
load_and_show('../input/train_noisy', samp_n.fname.values[0])


# #### 3 labels conditions

# In[ ]:


label_set = set(train.loc[(train.n_label == 3) & (train.is_curated == True), 'labels']) & set(
    train.loc[(train.n_label == 3) & (train.is_curated == False), 'labels'])

label_samp = np.random.choice(list(label_set), 1)[0]
samp = train[(train.labels == label_samp) & (train.is_curated == True)].sample(1)
print('File name:', '../input/train_curated/{}'.format(samp.fname.values[0]), '\nLabel:', label_samp)
IPython.display.display(ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0])))
load_and_show('../input/train_curated', samp.fname.values[0])


# In[ ]:


# sampling an audio in train_noisy
samp_n = train[(train.labels == label_samp) & (train.is_curated == False)].sample(1)
print('File name:', '../input/train_noisy/{}'.format(samp_n.fname.values[0]), '\nLabel:', samp_n.labels.values[0])
IPython.display.display(ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0])))
load_and_show('../input/train_noisy', samp_n.fname.values[0])


# ### 4 and more labels
# 
# There are no common record with same multi labels in curated and noisy. Will check audios in different subsets.

# In[ ]:


samp = train[(train.n_label == 4) & (train.is_curated == True)].sample(1)
print('File name:', '../input/train_curated/{}'.format(samp.fname.values[0]), '\nLabel:', samp.labels.values[0])
IPython.display.display(ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0])))
load_and_show('../input/train_curated', samp.fname.values[0])


# In[ ]:


samp = train[(train.n_label == 4) & (train.is_curated == False)].sample(1)
print('File name:', '../input/train_noisy/{}'.format(samp.fname.values[0]), '\nLabel:', samp.labels.values[0])
IPython.display.display(ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0])))
load_and_show('../input/train_noisy', samp.fname.values[0])


# In[ ]:


samp = train[(train.n_label == 5) & (train.is_curated == False)].sample(1)
print('File name:', '../input/train_noisy/{}'.format(samp.fname.values[0]), '\nLabel:', samp.labels.values[0])
IPython.display.display(ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0])))
load_and_show('../input/train_noisy', samp.fname.values[0])


# In[ ]:


samp = train[(train.n_label == 6) & (train.is_curated == False)].sample(1)
print('File name:', '../input/train_noisy/{}'.format(samp.fname.values[0]), '\nLabel:', samp.labels.values[0])
IPython.display.display(ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0])))
load_and_show('../input/train_noisy', samp.fname.values[0])


# Summary
# 
# * Samples with multi labels sound really difficult even for people to label all events correctly. Spectrograms of those samples maybe confused with all events happen the same time.
# 
# eg. "File name: ../input/train_noisy/5fde4352.wav Label: Fill_(with_liquid),Water_tap_and_faucet,Hiss,Toilet_flush,Sink_(filling_or_washing)"
# 
# * **A lot of noisy data seem to have wrong labels**. In the previous competition, using the robust loss function to suppress the effect of mislabeled data was one of the important points to get high score. We must care of that again.

# ### Co-occurrence
# 
# How to get labels for the trainning process.
# 
# ref:
# * https://www.kaggle.com/maxwell110/explore-multi-labeled-data

# In[ ]:





# In[ ]:




