#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import re
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


os.getcwd()


# In[ ]:


os.chdir('/kaggle/input/chai-time-data-science/')


# In[ ]:


thumbnail_types = pd.read_csv('Anchor Thumbnail Types.csv')
thumbnail_types.head()


# In[ ]:


thumbnail_types.shape


# ### This concludes Thumbnail type has only 4 major categories

# In[ ]:


description = pd.read_csv('Description.csv')
description.shape


# In[ ]:


description.head()


# In[ ]:


# Total Episodes 
description['episode_id'].nunique()


# ## Segregating Description and links from description dataframe as well as cleaning Description of episodes

# In[ ]:


description['description'][1]


# In[ ]:


get_ipython().system(' pip install contractions')


# In[ ]:


import contractions


# In[ ]:


def extracting_links(text):
    text = re.sub(r'\n',' ',text)
    # Getting URLS present in the description of the episode
    l = re.findall(r'http[s]*://[a-zA-Z0-9./-]+',text)
    return l
    
    


# In[ ]:


description['links']= description['description'].apply(lambda x : extracting_links(x))


# In[ ]:


def cleaning_description(text):
    # Substituting \n with a space 
    text = re.sub(r'\n',' ',text)
    # Cleaning URLS present in the description of the episode
    text = re.sub(r'http[s]*://[a-zA-Z0-9./-]+','',text)
    # Converting text to lower case 
    text = text.lower()
    # Expanding Contractions eg. i'll to I will 
    text = contractions.fix(text)
    return text


# In[ ]:


description['description'] = description['description'].apply(lambda x : cleaning_description(x))


# In[ ]:


description.head()


# In[ ]:


youtube_thumbnail = pd.read_csv('YouTube Thumbnail Types.csv')
youtube_thumbnail.shape


# In[ ]:


youtube_thumbnail.head()


# In[ ]:


episodes = pd.read_csv('Episodes.csv')
episodes.shape


# In[ ]:


episodes.head()


# In[ ]:


episodes.columns


# ### Check for the NaN values in the episodes dataframe

# In[ ]:


episodes.isnull().sum()


# In[ ]:


episodes['heroes'].unique()


# ### Let's count for the gender of the heroes interviewed

# In[ ]:


sns.countplot(episodes['heroes_gender'])


# ### A count analysis on the heroes_nationality

# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(episodes['heroes_nationality'])


# ### Sorting Episodes by the number of youtube subscribes each heroes have

# In[ ]:


episodes_sorted = episodes.sort_values('youtube_subscribers',ascending=False).reset_index(drop= True)


# In[ ]:


plt.figure(figsize=(12,19))
sns.barplot(episodes_sorted['youtube_subscribers'],episodes_sorted['heroes'])


# ### Top Ten Heroes based on the number of Youtube Subscribers 

# In[ ]:


episodes_sorted['heroes'][:10]


# ### Count of heroes for each category

# In[ ]:


sns.countplot(episodes['category'])


# ### Analysing the category of the heroes gender-wise

# In[ ]:


sns.countplot(episodes['category'],hue=episodes['heroes_gender'])


# In[ ]:




