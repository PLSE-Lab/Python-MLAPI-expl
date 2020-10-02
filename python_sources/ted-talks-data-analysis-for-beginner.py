#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading data
ted = pd.read_csv('../input/ted_main.csv')


# In[ ]:


#recognizing features
ted.head()


# In[ ]:


#info about features and target
ted.info()

#there is no NaN value or not needed feature in this dataset.


# In[ ]:


# Checking shape
ted.shape

# it has 2550 columns and 17 rows


# In[ ]:


# find out what datatypes are
ted.dtypes
# It has objects and integers


# In[ ]:


# checking if there is any null values
ted.isna().sum()
# There is one columns which has 6 null values


# In[ ]:


#Lets find out (What kind of topics attract the maximum discussion and debate (in the form of comments?)
ted.sort_values(by='views')
 #df[['title', 'main_speaker', 'views', 'film_date']].sort_values('views', ascending=False)


# In[ ]:


# get person who has maximum comments and views find out later
ted.sort_values(by='comments',ascending=False)
#we can see that Richard Dawkins: Militant atheism has maximam comments and 4374792 views


# In[ ]:


ted.sort_values(by='views',ascending=False)
#highest  Ken Robinson has highest views 47227110 and it was translated in 60 languages


# In[ ]:


#Which months are most popular among TED and TEDx chapters, so converting film date and published date to datetime
ted['published_date'] = pd.to_datetime(ted['published_date'],unit='s')


# In[ ]:


#Which months are most popular among TED and TEDx chapters
# ted['popular_month']=ted['published_date'].apply(lambda x:x[3:5])
# ted.head()
ted.sort_values(by='languages',ascending=False)# Matt Cutts's speech was tranlated into 72 languages


# In[ ]:


ted['published_date'].duplicated().value_counts()


# In[ ]:


ted['published_date'] = pd.to_datetime(ted['published_date'],unit='s')


# In[ ]:


ted['published_date'].head()


# In[ ]:


ted['film_date'] = pd.to_datetime(ted['film_date'],unit='s')
ted['film_date'].head()


# In[ ]:


ted.head()


# In[ ]:


# what was the first ted tallks
ted['film_date'].sort_values().head()
# we can see that 686 talks happend in 1972-05-14


# In[ ]:


# lets go deeper which one was the first talk
ted[ted['film_date']== '1972-05-14']


# The first talk film_date was 1972-05-14 and published_date was 2010-05-14 

# In[ ]:


# let's find out which one was the last talk
ted.groupby('published_date').sum().tail()


# we can see that 2017-09-22 15:00:22	last movie was relesed 

# In[ ]:


# now we can use this date to find out name of the movie
ted[ted['published_date']== '2017-09-22 15:00:22']


# so the  last ted talk name was Karoliina Korppoo: How a video game might help..

# That's it. Thank you for reading my notebook.

# In[ ]:




