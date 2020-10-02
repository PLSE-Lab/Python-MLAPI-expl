#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In order to get a better understanding of the data we are dealing with, and ultimately how it represents the reality we are modeling, there are a lot of procedures we can carry on to unveil its secrets. We are gonna use Pandas (https://pandas.pydata.org/pandas-docs/stable/) Pandas consists in two main data structures: Dataframe and Series, the first composed by one or many of the second.
# 
# To load a dataset as a dataframe:

# In[4]:


df = pd.read_csv("../input/moodle.csv.txt")
df.describe()


# In[5]:


#Take a look at the first rows of the dataset
df.head()


# In[8]:


#List all different values for storypoints
story_points_series = df['storypoint']
story_points_series.unique()


# **Visualization**

# 

# In[9]:


import matplotlib.pyplot as plt
story_points_series.plot.hist(figsize=(20,10))


# What about the distrubution for story points under 40?

# In[10]:


story_points_series_lower = story_points_series[lambda x: x <= 40]
story_points_series_lower.plot.hist(figsize=(20,10))


# A very interesting data representation is boxplot, to understand more about this check : https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

# In[11]:


story_points_series.plot.box(figsize=(20,10))


# **Word Cloud**
# 
# lets take a look on the title text data

# In[14]:


from wordcloud import WordCloud

text = " ".join(t for t in df.title) 

def create_word_cloud(text_data):
  wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(text_data)
  plt.figure(figsize=(12,10))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.show()

create_word_cloud(text)

