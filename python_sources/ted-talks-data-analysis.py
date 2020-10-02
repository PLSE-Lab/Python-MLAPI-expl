#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")


# Import Dataset

# In[3]:


tedt = pd.read_csv('../input/ted_main.csv')


# In[4]:


#read dataset
tedt.head()


# Explore Dataset

# In[5]:


print(tedt.info(verbose=False))


# This dataset has 2550 entries with 17 columns.

# Explore missing Values

# In[6]:


import missingno as msno
msno.matrix(tedt.sample(500))


# There is no missing value and the data is already cleaned as it is imported from TED website directly. One of the speaker_occupation is missing, although that can be ignored, keeping in mind it wont contribute much to the analysis.

# In[7]:


#import plot libraries for interacting visuals
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
# For Notebooks
init_notebook_mode(connected=True)
# For offline use
cf.go_offline()


# Views can indicate popularity of videos. So, we wil try to sort values by views to find out mostly viewed videos.

# In[8]:


mostviews = tedt[['title', 'main_speaker','speaker_occupation','views', 'languages', 'duration']].sort_values('views', ascending=False)
mostviews.head()


# These are the top 5 most watched videos. 

# Translation into different languages can increase number of view but it also depends on topic. As we can see below one of the talk is translated into 72 languages but it is not the mostly viewed video.

# In[9]:


tedt[tedt['languages']==72]


# To analyse it further we can check how views are distributed based on languages

# In[10]:


tedt['languages'].iplot(kind='hist', xTitle='Number of Languages', yTitle='Number of Talks')


# As we can see from the above plot, most of the talks are translated into 10-45 languages with 152 languages being translated into 26 languages. Although there are some outliers like 86 talks were not translated into any of the language, while some were trasnlated into more than 50 languages. We will explore number of views based on number of languages translated.

# In[11]:


mostviews.iplot(kind='scatter', x='languages', y='views', mode='markers')


# Above plot shows us that, more the number of languages a video/talk has been translated into, more the number of views the video got. Although, there can be other contributing factors as well. Here, as we can see that some of the videos are translated into more than 60 languages but they are not the most popular videos (in term. This implies that there can be some languages that are more spoken by people but some are not. Also, topic of the talk, speaker's way to deliver the speech, duration , his ability to engage audience can be some of the most contributing factors in increasing the popularity of the video.  
# 

# In[13]:


publish_date = pd.to_datetime(tedt["published_date"], unit='s').dt.year


# In[14]:


popularity = pd.DataFrame(tedt.groupby(publish_date)['views', 'languages', 'comments'].mean())
popularity


# In[15]:


tedt['speaker_occupation'].value_counts().head()


# In[ ]:





# In[ ]:




