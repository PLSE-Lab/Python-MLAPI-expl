#!/usr/bin/env python
# coding: utf-8

# **This notebook is meant to upload and visually explore the different classes of sounds in the FreeSound dataset**
# 
# Audio samples (i.e. a sound) can be very different from each other, even if they are said to be the same class of sound (e.g. a drum). But can we get the sounds, in a given audio class, that are the most representative or most common in an audio class? Of course sounds won't be exactly the same, but there are sounds more like each other than the rest. 
# 
# This notebook explores the audio data by visualizing the frame rate of different sounds in the dataset, like trumpet, bird chirps, and drums. Then, the most representative sounds from each audio class. 
# 
# By visualizing the frame rate of different sounds, we can get a sense how the audio samples and the audio classes are distinct from each other. In other words, we can visually inspect certain features between samples of the audio classes. If the audio classes are distinct, we have confidence in classifying them. If they are not very distinct, then we have to do a sophisticated approach to classify. 

# load libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# upload data

# In[3]:


train = pd.read_csv("../input/train.csv")
print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.label.unique()))
test = pd.read_csv("../input/sample_submission.csv")


# In[4]:


_,ax=plt.subplots(figsize=(7,7))
data = train.groupby(['label']).size().sort_values()
p = data.plot(kind='barh',ax=ax)
p = ax.set_xlabel("Count")
p = ax.set_ylabel("Audio class")


# The audio classes are non-uniformly distributed, for more than majority of the classes.

# example sound

# In[5]:


import IPython.display as ipd  # To play sound in the notebook
fname = '../input/audio_train/' + '00044347.wav'   # Hi-hat
ipd.Audio(fname)


# I don't care to listen to the individual sounds but I care about seeing the sounds for the whole class. I'd like to get a representative picture for each class of sounds. Hearing that average class sound will mean nothing to us...but if we can see that there might be distinctive properties between the different classes, that might help us to decide on a classifier.
# 
# Now, I need to be able to correctly grab the audio files of particular classes. I first do this for one example class, and then extrapolate to all the other classes. I'm sorting the file names by the order found in the train csv file. This is useful so that I can quickly grab files by their class

# In[6]:


import os


# In[7]:


train_files = os.listdir("../input/audio_train")
train_files_dict = dict(zip(train_files,range(len(train_files))))
sorted_train = train.iloc[train.fname.map(train_files_dict)]


# In[8]:


import ipywidgets as widget


# In[9]:


labels = train['label'].unique()
w = widget.Dropdown(options = labels, value=labels[0])


# In[10]:


w


# In[11]:


train_files_inds = train['label'] == w.label


# These are the files associated to this sound class

# In[12]:


train['fname'][train_files_inds].head(5)


# Now I want to load one of these and show as a spectrogram

# In[13]:


from scipy.io import wavfile
path = "../input/audio_train/"
fname = train['fname'][train_files_inds].values[0]
rate, data = wavfile.read(path + fname)


# In[14]:


plt.plot(data, '-', );


# Now for n files of the classes.  I'm actually plotting for only 5 classes because it is a lot to plot

# In[15]:


seed = 2
np.random.seed(seed)


# In[16]:


show_df = train.query('manually_verified == 1').sort_values('label')
labels = show_df['label'].unique()

for label in labels[:5]:
    
    train_files_inds = show_df['label'] == label

    rand_inds = np.random.randint(0,show_df['fname'][train_files_inds].count(),5)
    fnames = show_df['fname'].iloc[rand_inds]

    _, axs = plt.subplots(figsize=(17,4),nrows=1,ncols=5)

    for i,fname in enumerate(fnames):
        rate, data = wavfile.read(path + fname)
        axs[i].plot(data, '-', label=fname)
        axs[i].legend()
    plt.suptitle(label,x=0.04,y=0.5,horizontalalignment='center', fontsize=20)
    del rate
    del data
del axs


# Awesome! Now I can see just 5 random audio frames for each class (manually verified only). 
# 
# But this kernel is for getting the average representative samples for each class. 
# 
# So how do I get the representative samples? I get the median (I bet the distribution is skewed) of the samples for each class and I pick the samples that are closest to this value. 
# 
# What do I mean by median of the class? 
# 
# An audio file is a sample. Each sample has 44100 frames per second of audio. Each of those frames has a different magnitude representing the "loudness". The higher the magnitude of the frame the higher the pitch. 
# 
# Within a class, a sample may have different numbers of frames. In order to get the most representative or median sample, I need to take the median at each corresponding frame. 
# 
# Let's say I have a class with 5 samples with the same size-all just 5 frames each, where at each frame in a sample there is a certain magnitude. 

# In[17]:


arrs = []
lens = [5]*5
for i in lens:
    arrs.append(np.random.randint(7,size=i))
arrs = np.array(arrs)
arrs


# Say the above are my five samples from an audio class. What's the median? 

# In[18]:


np.median(arrs,axis=1)


# That was easy! But what if the samples are of different lengths,

# In[19]:


arrs = []
lens = [5,10,7,6,9]
for i in lens:
    arrs.append(np.random.randint(7,size=i))
arrs = np.array(arrs)
arrs


# Taking the median doesn't work because it doesn't broadcast. I need to find the longest array, make an array of 0s of that length, and pad the other arrays

# In[20]:


maxarrn = np.max([len(i) for i in arrs])
padded_arrs = [np.pad(arr,(0,maxarrn-len(arr)),mode='constant')for arr in arrs]
padded_arrs = np.array(padded_arrs)
display(padded_arrs)

med_padded_arrs = np.median(padded_arrs,axis=0)
med_padded_arrs


# Groovy! But I want to choose actual samples that are close to the median array. This means I have to compute a distance. I use _np.linalg.norm_ to get the vector norm, which is a measure of the distance between two arrays. So I compute the distances of each sample to the median

# In[21]:


dists = [np.linalg.norm(np.pad(arr,(0,len(med_padded_arrs)-len(arr)),mode='constant')-med_padded_arrs) for arr in arrs]
dists = np.array(dists)
dists


# Then I choose which sample I want by ordering so I can get those samples with the smallest distance

# In[22]:


dists.argsort()[::-1]


# So I would like at the samples from the first indices in the above because they are closest to the median. Now what if I do this for a whole audio class

# In[23]:


label = 'Acoustic_guitar'
sub = show_df[show_df.label==label]
fnames = sub.fname.values
arrs = []
for i,fname in enumerate(fnames):
        rate, data = wavfile.read(path + fname)
        arrs.append(data)
maxarrn = np.max([len(i) for i in arrs])
padded_arrs = [np.pad(arr,(0,maxarrn-len(arr)),mode='constant')for arr in arrs]
padded_arrs = np.array(padded_arrs)
med_padded_arrs = np.median(padded_arrs,axis=0)
dists = [np.linalg.norm(np.pad(arr,(0,len(med_padded_arrs)-len(arr)),mode='constant')-med_padded_arrs) for arr in arrs]
dists = np.array(dists)
fnames = sub.iloc[dists.argsort()[::-1]].head(5).fname.values
_, axs = plt.subplots(figsize=(17,4),nrows=1,ncols=5)
for i,fname in enumerate(fnames):
    rate, data = wavfile.read(path + fname)
    axs[i].plot(data, '-', label=fname)
    axs[i].legend()
plt.suptitle(label,x=0.04,y=0.5,horizontalalignment='center', fontsize=20)


# The above are the 5 most representative (closest tov the median) for Acoustic_guitar. Now I can do this for all classes

# In[24]:


show_df = train.query('manually_verified == 1').sort_values('label')
labels = show_df['label'].unique()

for label in labels:
    sub = show_df[show_df.label==label]
    fnames = sub.fname.values
    arrs = []
    for i,fname in enumerate(fnames):
            rate, data = wavfile.read(path + fname)
            arrs.append(data)
    maxarrn = np.max([len(i) for i in arrs])
    padded_arrs = [np.pad(arr,(0,maxarrn-len(arr)),mode='constant')for arr in arrs]
    padded_arrs = np.array(padded_arrs)
    med_padded_arrs = np.median(padded_arrs,axis=0)
    dists = [np.linalg.norm(np.pad(arr,(0,len(med_padded_arrs)-len(arr)),mode='constant')-med_padded_arrs) for arr in arrs]
    dists = np.array(dists)
    fnames = sub.iloc[dists.argsort()[::-1]].head(5).fname.values
    _, axs = plt.subplots(figsize=(17,4),nrows=1,ncols=5)
    for i,fname in enumerate(fnames):
        rate, data = wavfile.read(path + fname)
        axs[i].plot(data, '-', label=fname)
        axs[i].legend()
    plt.suptitle(label,x=0.04,y=0.5,horizontalalignment='center', fontsize=20)
del axs


# There will be a follow up notebook exploring another aspect of the data by visualizing spectrograms.

# In[25]:


fname = '../input/audio_train/' + '991fa1d7.wav'   # Hi-hat
ipd.Audio(fname)


# In[ ]:




