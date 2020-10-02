#!/usr/bin/env python
# coding: utf-8

# **Import the necessary libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# **lets look at the given datasets**

# In[ ]:


ls ../input/chai-time-data-science/


# **There are 4 csv files and two directories. Lets check what is there in that directories**

# In[ ]:


ls ../input/chai-time-data-science/'Raw Subtitles'


# **It contain lots of text file, let's look at one of these text file and see what it is:**

# In[ ]:


with open('../input/chai-time-data-science/Raw Subtitles/E1.txt') as f:
    print(f.readlines())


# So, this is the whole conversation of every episodes. If we read few lines we can see it has a pattern: `name`,`time` and `text`. 

# **lets look `Cleaned Subtitles` directory:**

# In[ ]:


ls ../input/chai-time-data-science/'Cleaned Subtitles'


# In[ ]:


temp = pd.read_csv('../input/chai-time-data-science/Cleaned Subtitles/E1.csv')
temp.head()


# **Woah!! I am amazed how did he prepared the dataset and how much time did he spend preparing the dataset.**

# # Here is the summary of whatever we've discovered so far:
# 
# * There are 4 CSV file named : `Anchor Thumbnail Types.csv`, `Description.csv`, `Episodes.csv` and `YouTube Thumbnail Types.csv`.
# * Apart from these 4 CSV file there are two directory : `Raw Subtitles` and `Cleaned Subtitles`.
# * `Raw Subtitles` containg 85 raw text file which is conversation in each episodes.
# * `Cleaned Subtitles` contain 85 csv file with column name `Time`, `Speaker` and `text`. At time t who is the speaker and what is he saying in the conversation.

# # Let's look at the CSV files 

# In[ ]:


# reading the datasets
DATA_DIR = '../input/chai-time-data-science/'
desc = pd.read_csv(DATA_DIR+'Description.csv')
epsd = pd.read_csv(DATA_DIR+'Episodes.csv')
yt_th = pd.read_csv(DATA_DIR+'YouTube Thumbnail Types.csv')
an_th = pd.read_csv(DATA_DIR+'Anchor Thumbnail Types.csv')


# In[ ]:


# lets first look at description 
desc.head()


# This file contain episode id and descriptions. Lets check what is the total number of episode and few description.

# In[ ]:


print(desc.shape)


# Looks like every row contain unique episode and their description. So there are 85 unique episodes.

# In[ ]:


episode_id, describe = desc.loc[0]
print(episode_id)
print('---------------------------------------')
print(describe)


# In[ ]:


episode_id, describe = desc.loc[2]
print(episode_id)
print('---------------------------------------')
print(describe)


# In[ ]:


episode_id, describe = desc.loc[10]
print(episode_id)
print('---------------------------------------')
print(describe)


# # Lets look another dataframe

# In[ ]:


epsd.head()


# In[ ]:


epsd.shape


# In[ ]:


epsd.columns


# **This file contain lots of detils like : episodes info, heros detail, youtube detail, something called anchor(I don't know what it is, we will look at it later), and spotify detail. We will look at this dataset later. We first look at eash readable file**

# # lets look at another dataframe

# In[ ]:


# youtube thumbnail 
yt_th


# **The name of this dataset is youtube thumbnail type, I don't know what it is. Lets search on google :** 
# 
# Here is something I got: YouTube thumbnail is the short sequence of photos played on youtube when you put curse on the video.
# 
# I don't understand what is in this dataset and how it is useful. Lets look at another dataframe.

# In[ ]:


an_th.head()


# **The name of this dataset is Anchor thumbnail Type, I don't know what it is. Lets search on google :** 
# 
# Here is something I got: It the similar like we have seen abouve but it it used in podcast.

# # Let's recap what we have done so far: 
# 
# * We have given 4 csv file and two directory. Each directory contain the conversation with each guest with text and csv formated.
# * Amoung 4 csv file two are the thumbnail detail(Which I don't know how to use yet). 
# * One csv file contain the episode id, and a short description on each episode.
# * So, the remain one named **episode.csv**, which contain lots of information. 
# * Now, lets go a little bit more deeper into this data

# In[ ]:


epsd.info()


# **Since it has lots of columns we mush summerise is so that It will be easy to target perticular column:** 
#  
# |Column No | About |
# |----------|-------:
# |0-1        |episode detail|
# |2-7        | About Heroes|
# |9          | Chai |
# |10-11|recording date and time|
# |14-26      |About YouTube|
# |27, 28, 29 |About Anchor |
# |30,31,32   | About Spotify|
# |33,34,35   | Apply        |

# ### It's time to have come curiosity, without it we can't do anything. 
# 
# ### I will generate some question I want to know the answer. While solving those questions if any other arises we will try to find the answer of that. 
# 
# ### To generate questions I will look of the table we just created:
# 
# ### Here are some of the question I am curious about:
# 
# 1. What are the total number of episode.
# 1. Total number of unique heros and some information about them.
# 1. What are the total number of unique chai, most fevourite chai.
# 1. Recording frequency
# 1. Youtube information like : total number of like, cumment etc and same for other platform.

# Since I am not an expert in analysis, It will be better to write all my weapons of analysis which I will collect from kaggle learn and pandas documentation, so it will be easy to decice which to use:
# 
# |Type of chart| requirement data types|When to use
# |-------------|:------------:|-------------------:
# |Linechart|One or more continuous vectors(array)|
# |Barchart|Categorical vs continuous|
# |Heatmap|One or more continuous vectors|
# |Scatter Plot|continuous vs continuous|to find relation between two continuou variable
# |Hiotogram|single continuous vector|To see the destribution of a variable
# |Pie|
# |Kde plot|

# I will try to complete this table and add some more element when I will learn during this analysis. Let's get start with whatever we have now.

# **Question 1: What are the total number of episodes?**

# In[ ]:


epsd.episode_id.head()


# In[ ]:


epsd.episode_id.nunique()


# Answer: There are 85 unique episodes.

# **Question 2 : Total number of unique heros and some information about them**

# In[ ]:


# heros columns
heros = epsd.iloc[:,2:8]   # select the whole rows and column no 2-7(including)
heros.head()


# **Why there are NaN in zeroth row??**
# 
# Let's see the whole zeroth row of episode dataframe.

# In[ ]:


epsd.loc[0]


# looks like there is no heros in episode zero, if we look at episode name its `chai time data science announcement` and thats is the reason behind our heros row in NaN. Okey let's move to our point.

# In[ ]:


# unique heros
heros.heroes.describe()


# So, we have 74 total heros and 72 unique. and 'Shivam Bansal' came two times. But there are 85 episodes. Which meand there are some more columns apart from episode E0 where no heros came.
# 
# **What are the total number of such episodes where no heros came?**

# In[ ]:


heros.heroes.isna().sum()


# There are 11 such episode where no heros came. Let's look at those episode.

# In[ ]:


no_hero_epsd = epsd.loc[epsd.heroes.isna()==True]
no_hero_epsd.loc[:,['episode_id','episode_name']]


# * **E0 : Announcement| M0-M8(total=9) : About Fastai | E78 : Birthday**

# Lets get back to know about heros.

# In[ ]:


heros.head()


# In[ ]:


# heros gender ration
heros.heroes_gender.value_counts()


# In[ ]:




