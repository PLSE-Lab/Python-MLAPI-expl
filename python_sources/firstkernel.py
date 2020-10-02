#!/usr/bin/env python
# coding: utf-8

# # Analysis of Trending Youtube Videos in 5 Countries
# 
# 
# This is my first attempt at a kernel/notebook so I apologize for any mistakes. Countries used in analysis are USA, Great Britain, Germany, Canada, and France

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Reading the data for 5 countries (USA, Great Britain, Germany, Canada, and France) from given csv files:

# In[ ]:


dataFR = pd.read_csv('/kaggle/input/youtube-new/FRvideos.csv')
dataCA = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv')
dataUS = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
dataGB = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv')
dataDE = pd.read_csv('/kaggle/input/youtube-new/DEvideos.csv')


# To proceed, an example data is inspected below to get an idea of its elements:

# In[ ]:


dataCA.head()


# Collecting more information on data walso gives intiutive predictions about what can be inspected, for instance correlation between likes and comments can be examined in the future.

# In[ ]:


dataCA.info()
dataFR.info()


# There are 16 columns in data, 
# 
# * 3 of those are boolean variables , 
# * 5 of them are integer variables 
# * 8 of them are object/string variables.
# 
# Total number of videos slightly differs between countries. We can see that there are 40881 videos in Canadian trending videos and  40724 in French trending videos. 

# # Combining Country Data

# In[ ]:


dataALL = pd.concat([dataCA,dataFR,dataUS,dataDE,dataGB],axis =0)


# In[ ]:


dataALL.info()


# We have combined our datas using pandas concat. This way in the future if we want to analyze something and do not want country bias and want to look at the bigger worldwide picture we can use our dataALL.

# # Correlations

# In[ ]:


dataALL.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(dataALL.corr(), annot=True, linewidths=.9, fmt= '.2f',ax=ax, cbar=True)
plt.show()


# From this heatmap we can infer that there is a high correlation between comment counts and likes/dislike. 
# * Correlation between comment count and likes : **0.780923**
# * Correlation between comment count and dislikes : **0.727815**
# 

# In[ ]:


print(dataALL.columns)
print(dataALL.nunique())


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
dataALL.likes.plot(color = 'b',label = 'Likes',linewidth=1, alpha = 0.4,grid = True,linestyle = '-.')
dataALL.dislikes.plot(color = 'red',label = 'Dislikes',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot 1: Likes vs Dislikes')            # title = title of plot
plt.show()


# In[ ]:


dataALL.comment_count.plot(kind = 'line', color = 'r',label = 'Comment count',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
dataALL.likes.plot(color = 'b',label = 'Likes',linewidth=1, alpha = 0.4,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot 2: Comment Count vs Likes')            # title = title of plot
plt.show()


# In[ ]:


dataALL.comment_count.plot(kind = 'line', color = 'r',label = 'Comment count',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
dataALL.dislikes.plot(color = 'b',label = 'Dislikes',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot 3: Comment Count vs Dislikes')            # title = title of plot
plt.show()


# As it can be seen some videos are contreversial where they have significant amount of comments, likes and dislikes.

# In[ ]:


dataALL.plot(kind='scatter', x='comment_count', y='dislikes',alpha = 0.5,color = 'red')
plt.xlabel('comment_count')              # label = name of label
plt.ylabel('dislikes')
plt.title('Comment Count - Dislikes Scatter Plot')            # title = title of plot


# In[ ]:


dataALL.plot(kind='scatter', x='likes', y='views',alpha = 0.5,color = 'purple')
plt.xlabel('likes')              # label = name of label
plt.ylabel('views')
plt.title('Comment Count - Dislikes Scatter Plot')            # title = title of plot


# While it concentrates in low amount of comments and dislikes, it is obvious from the scatter graph that there is correlation between comments counts and dislikes. We can interpret this as people tend to leave relatively more comments more if they dislike a video.Lets continue with inspecting views, likes, dislikes and comment counts and check their distributions

# # Distribution

# In[ ]:


dataALL['likes_log'] = np.log(dataALL['likes'] + 1)
dataALL['views_log'] = np.log(dataALL['views'] + 1)
dataALL['dislikes_log'] = np.log(dataALL['dislikes'] + 1)
dataALL['comment_log'] = np.log(dataALL['comment_count'] + 1)

plt.figure(figsize = (12,6))

plt.subplot(221)
g1 = sns.distplot(dataALL['views_log'])
g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)

plt.subplot(224)
g2 = sns.distplot(dataALL['likes_log'],color='green')
g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)

plt.subplot(223)
g3 = sns.distplot(dataALL['dislikes_log'], color='r')
g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)

plt.subplot(222)
g4 = sns.distplot(dataALL['comment_log'])
g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()


# These all 4 graphs look similar to a normal distribution.

# In[ ]:


# quintiles of data

print("Views quantiles")
print(dataALL['views'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Likes quantiles")
print(dataALL['likes'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Dislikes quantiles")
print(dataALL['dislikes'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Comment quantiles")
print(dataALL['comment_count'].quantile([.01,.25,.5,.75,.99]))


# # Some filtering with Pandas

# In[ ]:


x = dataALL['views']>300000000    # videos with more than 300 million views
dataALL[x].title.unique()


# There were 3 videos with more than 300 million views, I had to also filter them as unique because we are using 5 country data same videos were coming as a result multiple times.

# In[ ]:


yy = dataALL[np.logical_and(dataALL['views']>300000000, dataALL['dislikes']<200000 )]
yy.title.unique()
#videos with videos with more than 300 million views and less than 200 000 dislikes


# In[ ]:




