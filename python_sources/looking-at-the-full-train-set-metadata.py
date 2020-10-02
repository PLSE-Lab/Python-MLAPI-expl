#!/usr/bin/env python
# coding: utf-8

# ### This notebook explores what we have in the DFDC full train set metadata and json files.

# The dataset for this notebook  is at
# https://www.kaggle.com/zaharch/train-set-metadata-for-dfdc
# 
# The train data for this competition is big, almost 500Gb, so I hope it can be useful to have all the json files and the metadata in one dataframe.
# 
# The dataset includes, for each video file
# 1. Info from the json files: **filename**, **folder**, **label**, **original**
# 2. **split**: train (118346 videos), public validation test (400 videos) or train sample (400 videos). 119146 videos in total. Note that the public validation and the train sample are subsets of the full train, so it is enough to mark them in this dataframe.
# 3. Full file **md5** column
# 4. Hash on audio file sequence **wav.hash** and on subset of pixels **pxl.hash**
# 5. The rest are metadata fields from the files, obtained with ffprobe. Note that I removed many columns, which didn't give new information.

# In[ ]:


import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[ ]:


data = pd.read_csv('/kaggle/input/train-set-metadata-for-dfdc/metadata', low_memory=False)


# # Hashes

# Fakes always have at least some pixel-level changes. That means that all audio fakes are also video fakes.

# In[ ]:


(data['pxl.hash'] == data['pxl.hash.orig']).value_counts()


# There are duplicated for both **md5**, **pxl.hash** and **wav.hash**. Duplicates for **wav.hash** are OK, but duplicates for **md5** mean that there are identical files in the dataset.

# In[ ]:


data['md5'].value_counts().value_counts().head()


# In[ ]:


data['wav.hash'].value_counts().value_counts().head()


# In[ ]:


data['pxl.hash'].value_counts().value_counts().head()


# # Other fields

# This is how the data looks like

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.label.value_counts()


# In[ ]:


data.split.value_counts()


# In[ ]:


set(data.original) - set(data.filename)


# In[ ]:


set(data.loc[data.original == 'NAN', 'filename']) - set(data.original)


# In[ ]:


data.loc[data.original != 'NAN', 'original'].value_counts().hist(bins=40)


# In[ ]:


data.loc[data.original != 'NAN', 'original'].value_counts().value_counts().head()


# In[ ]:


for col in data.columns:
    print(pd.crosstab(data[col],data['label']))


# In[ ]:


pd.crosstab(data['video.@display_aspect_ratio'],data['label'])


# In[ ]:


pd.crosstab([data['video.@display_aspect_ratio'], data['audio.@codec_time_base']],data['label'])


# In[ ]:




