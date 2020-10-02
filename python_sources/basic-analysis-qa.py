#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# This is my first kernel, I am just learning to get things work. If you find any mistake or suggestion feel free to add it into comments. It made me learn hope it does for you too..

# **First thing is to ask questions, like what to do with this dataset, what do i wanna do with it, what can be gained from this dataset.**

# Questions in this dataset would be 
# 1. How many of them mentioned Shahid kapoor or Kiara advani or where they only talking about the movie?
# 2. At what time do most of the tweets happen?
# 3. How many people make use of hastag feature?
# 
# 

# **Tasks can be**
# 1. Loading Dataset
# 1. Clean the data inside
# 1. Answer your questions

# In[ ]:


data = pd.read_csv("../input/twitter_kabir_singh_bollywood_movie.csv")
data.head()


# Let's check correlation between datasets' collumns.

# In[ ]:


data.corr() 


# Not much.

# In[ ]:


data.nunique()


# We have almost 12,596 people's data.

# We can see 'time' isn't unique, so we cannot find the time they tweeted 
# and 'id, type, retweeter, lang' are preety much useless. So we can remove them.

# In[ ]:


del data['re_tweeter']
del data['time']
del data['type']
del data['lang']
del data['id']


# **Now to find out how many are talking about shahid kapoor or kiara advani?
#  Which are @shahidkapoor @Advani_Kiara values in text_raw feature.**

# In[ ]:


temp_data = data
p = temp_data.text_raw.apply(lambda temp_data: temp_data.find('@shahidkapoor'))
print(p.nunique())
p = temp_data.text_raw.apply(lambda temp_data: temp_data.find('@Advani_Kiara'))
print(p.nunique())



# Removing 1 value which is -1(is returned when no value is found) We get 
# 1. Shadid Kapoor 268.
# 1. Kiara Advani 260.
#  
# So remainnig 12068 were talking something else i guess..

# **How many people make use of hastag feature?**

# In[ ]:


# data['hashtags'].isnull() this doesnt gives you the answer, because empty ones have atleast '[]' in them
temp_data = data
temp_data.hashtags = temp_data.hashtags.apply(lambda temp_data: temp_data.find('[]'))
print(temp_data.hashtags.value_counts())


# Which shows 9251 do and 3345 don't. Lets plot this.

# In[ ]:


hashtag_plt = temp_data.hashtags.astype(str)
hashtag_plt = hashtag_plt.replace('-1', 'Yes')
hashtag_plt = hashtag_plt.replace('0', 'No')
hashtag_plt.value_counts()


# In[ ]:


values = ["Yes", "No"]
index = [0,1]
plt.rcParams['figure.figsize'] = (3, 4)
plt.style.use('_classic_test')
hashtag_plt.value_counts().plot.bar(color = 'red')
plt.xlabel('Hashtag Features', fontsize = 15)
plt.ylabel('Hastag_users', fontsize = 15)
plt.xticks(index, values, rotation=0)
plt.show()


# Looks Good..

# That's it, I don't find any more questions for this data(apart from making a pie chart for people who talk about shahid,kiara,kabirsignh..more from text feature).
