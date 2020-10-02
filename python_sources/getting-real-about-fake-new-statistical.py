#!/usr/bin/env python
# coding: utf-8

# This dataset is taken from [Megan Risdal's Getting Real about Fake News](https://www.kaggle.com/mrisdal/fake-news).
# She says *"The latest hot topic in the news is fake news and many are wondering what data scientists can do to detect it and stymie its viral spread. This dataset is only a first step in understanding and tackling this problem. It contains text and metadata scraped from 244 websites tagged as "bullshit" by the BS Detector Chrome Extension by Daniel Sieradski.
# Warning: I did not modify the list of news sources from the BS Detector so as not to introduce my (useless) layer of bias; I'm not an authority on fake news. There may be sources whose inclusion you disagree with. It's up to you to decide how to work with the data and how you might contribute to "improving it". "*
# You can find out details on her dataset. 
# So i decided to explain more.
# 

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/fake.csv")


# In[ ]:


data.info()


# In[ ]:


data.corr()


# It gives us correlation between futures. It gets value between -1 and 1. If its close to 0, it means no connection about these two veriable. If its close to +1, it means positive correlation, so these two have direct proportion. If its close to -1,it means negative correlation, so these two have inverse proportion.   
# 

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.columns


# Coloumns;
# uuid => Unique identifier,
# ord_in_thread,
# author => author of story,
# published => date published,
# title => title of the story,
# text => text of story,
# language => data from webhose.io,
# crawled => date the story was archived,
# site_url => site URL from BS detector,
# country => data from webhose.io,
# domain_rank => data from webhose.io,
# thread_title,
# spam_score => data from webhose.io,
# main_img_url => image from story,
# replies_count => number of replies,
# participants_count => number of participants,
# likes => number of Facebook likes,
# comments => number of Facebook comments,
# shares => number of Facebook shares,
# type => type of website (label from BS detector)

# ### MATPLOTLIB
# Matplot is a python library that help us to plot data. The easiest and basic plots are line, scatter and histogram plots.
# * Line plot is better when x axis is time.
# * Scatter is better when there is correlation between two variables
# * Histogram is better when we need to see distribution of numerical data.
# * Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle  

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.likes.plot(kind = 'line', color = 'g',label = 'likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.shares.plot(color = 'r',label = 'shares',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='participants_count', y='replies_count',alpha = 0.5,color = 'red')
plt.xlabel('Participants')              # label = name of label
plt.ylabel('Replies')
plt.title('Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.spam_score.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.title("Spam Score")
plt.show()


# ### PANDAS
# What we need to know about pandas?
# * CSV: comma - separated values
# 
# 

# In[ ]:


data = pd.read_csv('../input/fake.csv')


# In[ ]:


series = data['spam_score']        # data['spam_score'] = series
print(type(series))
data_frame = data[['spam_score']]  # data[['spam_score']] = data frame
print(type(data_frame))


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['spam_score']==0     # There are 12999 rows that have 0 value for spam_score
data[x]


# In[ ]:


# 2 - Filtering pandas with logical_and
# There are 9619 rows that have 0 value for spam_score and 1 value for participants_count
data[np.logical_and(data['spam_score']==0, data['participants_count']==1 )]

