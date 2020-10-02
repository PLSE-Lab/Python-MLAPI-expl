#!/usr/bin/env python
# coding: utf-8

# 
# Thanks to @rohanrao, because now we have additional datasets for training:
# 
# https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-a-m <br>
# https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-n-z
# 
# But there are 14 columns missing, so we need to get somehow the data to fill them.<br> 
# 
# Let's check how we can do it. Hope it will be useful.

# In[ ]:


get_ipython().system('pip install tinytag')


# In[ ]:


get_ipython().system('pip install mutagen')


# In[ ]:


import numpy as np
import pandas as pd

import mutagen
from mutagen.mp3 import MP3
from tinytag import TinyTag


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# I found two nice libraries which can extract metadata from mp3:
# 
# ### MUTAGEN ###
# 
# https://pypi.org/project/mutagen/
# 
# ### TINY TAG ###
# 
# https://pypi.org/project/tinytag/
# 
# And in this notebook I used both of them.

# In[ ]:


DATA_PATH = '../input/birdsong-recognition/'
AUDIO_PATH = "../input/birdsong-recognition/train_audio"

DATA_PATH_EXT_A_M = '../input/xeno-canto-bird-recordings-extended-a-m/'
AUDIO_PATH_EXT_A_M = "../input/xeno-canto-bird-recordings-extended-a-m/A-M"

DATA_PATH_EXT_N_Z = '../input/xeno-canto-bird-recordings-extended-n-z/'
AUDIO_PATH_EXT_N_Z = '../input/xeno-canto-bird-recordings-extended-n-z/N-Z'


# Let's compare new datasets with old one

# In[ ]:


df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_train_ext_a_m = pd.read_csv(DATA_PATH_EXT_A_M + 'train_extended.csv')
df_train_ext_n_z = pd.read_csv(DATA_PATH_EXT_N_Z + 'train_extended.csv')


# In[ ]:


#Original dataset
for row in df_train.columns:
    print(row)


# In[ ]:


#New dataset from A to M
for row in df_train_ext_a_m.columns:
    print(row)


# In[ ]:


#New dataset from N to Z
for row in df_train_ext_n_z.columns:
    print(row)


# In[ ]:


a = df_train.columns
b = df_train_ext_a_m.columns
c = df_train_ext_n_z.columns


# In[ ]:


set(a).difference(b)


# In[ ]:


#Just to make sure that there is no difference between A-M and N-Z parts
set(b).difference(c)


# So, as we can see 14 columns are missing.

# ### Mutagen ###
# With Mutagen we can get data for columns: bitrate_of_mp3, length, channels. Let's do it for one file.

# In[ ]:


#Common info
mutagen.File("../input/xeno-canto-bird-recordings-extended-a-m/A-M/aldfly/XC133197.mp3")


# In[ ]:


#Only what we need
audio = MP3("../input/xeno-canto-bird-recordings-extended-a-m/A-M/aldfly/XC133197.mp3")
print(audio.info.bitrate)
print(audio.info.length)
print(audio.info.channels)


# ### TinyTag ###
# 
# TinyTag alows to extract data for columns: background, description, primary_label, sampling_rate, secondary_labels, title. Most of the data stored at tag.comment, so it is necessary to work with strings a little bit. By the way there are more useful tags for extraction, which you can find in docs.

# In[ ]:


tag = TinyTag.get("../input/xeno-canto-bird-recordings-extended-a-m/A-M/aldfly/XC133197.mp3", image=True)


# In[ ]:


print('file comment', tag.comment)
print('samples per second', tag.samplerate)
print('title of the song', tag.title)


# ### Rest columns ###
# 
# The rest columns are: number_of_notes, pitch, rating, speed, volume. Let's check what values they have.

# In[ ]:


df_train['volume'].value_counts()


# In[ ]:


df_train['number_of_notes'].value_counts()


# In[ ]:


df_train['pitch'].value_counts()


# In[ ]:


df_train['rating'].value_counts()


# In[ ]:


df_train['speed'].value_counts()


# All of these remaining columns are secondary of importance. I believe that they can simply be filled with standard methods for filling data gaps. <br>
# 
# <strong>So here are methods, which you can use for metadata extraction straight from the file to get more training data or to use in some other way.</strong>
