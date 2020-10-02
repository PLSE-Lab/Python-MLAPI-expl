#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataFR = pd.read_csv("/kaggle/input/youtube-new/FRvideos.csv")
dataCA = pd.read_csv("/kaggle/input/youtube-new/CAvideos.csv")
dataUS = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
dataIN = pd.read_csv("/kaggle/input/youtube-new/INvideos.csv")
dataDE = pd.read_csv("/kaggle/input/youtube-new/DEvideos.csv")
dataGB = pd.read_csv("/kaggle/input/youtube-new/GBvideos.csv")


# In[ ]:


print("FRANCE:")
print(" ")
print("Information:")
print(" ")
dataFR.info()
dataFR.head()


# In[ ]:


print("CANADA:")
print(" ")
print("Information:")
print(" ")
dataCA.info()
dataCA.head()


# In[ ]:


print("UNITED STATES:")
print(" ")
print("Information:")
print(" ")
dataUS.info()
dataUS.head()


# In[ ]:


print("INDIA:")
print(" ")
print("Information:")
print(" ")
dataIN.info()
dataIN.head()


# In[ ]:


print("GERMANY:")
print(" ")
print("Information:")
print(" ")
dataDE.info()
dataDE.head()


# In[ ]:


print("GLOBAL:")
print(" ")
print("Information:")
print(" ")
dataGB.info()
dataGB.head()


# **In the dataframes, there are only 5 parameters are numerical, so I can only compare these attributes of the trend videos.**
# **Nevertheless, I can compare a string or boolean feature's numerical value, for example title's or channel_title's, with another one, but there are some data I don't use such as video_id, description. First, I want to gather all the dataframes in a list and then, get rid of the useless attributes one by one. Becaue I don't want to mix them, I use a list instead of concatenating them.**

# In[ ]:


data_frame_list = [dataFR,dataCA,dataUS,dataIN,dataDE,dataGB]


# In[ ]:


for data_frame in data_frame_list:
    data_frame.drop('video_id',axis=1,inplace=True)
    data_frame.drop('description',axis=1,inplace=True)
    data_frame.drop('thumbnail_link',axis=1,inplace=True)


# **Now, I can compare the attributes. But first, I want to be sure no item in a dataframe, which is special for a country, is also in the global one.**

# In[ ]:


data_frame_list[5]


# In[ ]:


same = []
for data_frame in data_frame_list:
    for each in data_frame['title']:
        if each in dataGB['title']:
            same.append(each)


# In[ ]:


print(same)


# In[ ]:


del data_frame_list


# **As you can see, there is no same item. Then, I can begin to visualize.**

# In[ ]:


print("FRANCE:")
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(dataFR.corr(), annot=True, linewidths=.5, fmt ='0.1f', ax=ax, cmap='Blues')
plt.show()


# In[ ]:


dataFR.corr()


# In[ ]:


dataFR.likes.plot(kind='line', color='r', label='Likes', linewidth=1, alpha=0.5, grid=True, linestyle=':')
dataFR.dislikes.plot(kind='line', color='b', label='Dislikes', linewidth=1, alpha=0.5, grid=True, linestyle='-')
plt.title('Trend Videos in France - Likes, Dislikes')
plt.legend(loc='upper left')
plt.xlabel('Song Index')
plt.ylabel('Value')
plt.show()


# In[ ]:


print("CANADA:")
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(dataCA.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax, cmap="Blues")
plt.show()


# In[ ]:


dataCA.corr()


# In[ ]:


dataCA.likes.plot(kind='line', color='r', label='Likes', linewidth=1, alpha=0.5, grid=True, linestyle=':')
dataCA.dislikes.plot(kind='line', color='b', label='Dislikes', linewidth=1, alpha=0.5, grid=True, linestyle='-')
plt.title('Trend Videos in Canada - Likes, Dislikes')
plt.legend(loc='upper left')
plt.xlabel('Song Index')
plt.ylabel('Value')
plt.show()


# In[ ]:


print("UNITED STATES:")
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(dataUS.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap="Blues")
plt.show()


# In[ ]:


dataUS.corr()


# In[ ]:


dataUS.likes.plot(kind='line', color='r', label='Likes', linewidth=1, alpha=0.5, grid=True, linestyle=':')
dataUS.dislikes.plot(kind='line', color='b', label='Dislikes', linewidth=1, alpha=0.5, grid=True, linestyle='-')
plt.title('Trend Videos in United States - Likes, Dislikes')
plt.legend(loc='upper left')
plt.xlabel('Song Index')
plt.ylabel('Value')
plt.show()


# In[ ]:


print("INDIA:")
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(dataIN.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap="Blues")
plt.show()


# In[ ]:


dataIN.corr()


# In[ ]:


dataIN.likes.plot(kind='line', color='r', label='Likes', linewidth=1, alpha=0.5, grid=True, linestyle=':')
dataIN.dislikes.plot(kind='line', color='b', label='Dislikes', linewidth=1, alpha=0.5, grid=True, linestyle='-')
plt.title('Trend Videos in India - Likes, Dislikes')
plt.legend(loc='upper right')
plt.xlabel('Song Index')
plt.ylabel('Value')
plt.show()


# In[ ]:


print("GERMANY:")
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(dataDE.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap="Blues")
plt.show()


# In[ ]:


dataDE.corr()


# In[ ]:


dataDE.likes.plot(kind='line', color='r', label='Likes', linewidth=1, alpha=0.5, grid=True, linestyle=':')
dataDE.dislikes.plot(kind='line', color='b', label='Dislikes', linewidth=1, alpha=0.5, grid=True, linestyle='-')
plt.title('Trend Videos in Germany - Likes, Dislikes')
plt.legend(loc='upper left')
plt.xlabel('Song Index')
plt.ylabel('Value')
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(dataGB.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap='Blues')
plt.show()


# In[ ]:


dataGB.corr()


# In[ ]:


dataGB.likes.plot(kind='line', color="r", label="likes", linewidth=1, alpha=0.5, grid=True, linestyle=":")
dataGB.dislikes.plot(kind='line', color="b", label="Dislikes", linewidth=1, alpha=0.5, grid=True, linestyle="-")
plt.title('Trend Videos in Global - Likes, Dislikes')
plt.legend(loc='upper left')
plt.xlabel('Song Index')
plt.ylabel('Value')
plt.show()


# **-------------Comprison mean of views of all countries---------------**

# In[ ]:


country=['France','Canada','United States','India','Germany','Global']
mean_views= [dataFR['views'].mean(),dataCA['views'].mean(),dataUS['views'].mean(),dataIN['views'].mean(),dataDE['views'].mean(),dataGB['views'].mean()]
plt.bar(country,mean_views)
plt.title('Mean of Views - Countries')
plt.show()


# **I wonder that is there any relation between dislike and error or removed videos.**

# In[ ]:


plt.scatter(dataFR.video_error_or_removed,dataFR.dislikes,color='blue',label='France',alpha=0.7)
plt.scatter(dataCA.video_error_or_removed,dataCA.dislikes,color='red',label='Canada',alpha=0.7)
plt.scatter(dataUS.video_error_or_removed,dataUS.dislikes,color='yellow',label='United States',alpha=0.7)
plt.scatter(dataIN.video_error_or_removed,dataIN.dislikes,color='purple',label='India',alpha=0.7)
plt.scatter(dataDE.video_error_or_removed,dataDE.dislikes,color='black',label='Germany',alpha=0.7)
plt.scatter(dataGB.video_error_or_removed,dataGB.dislikes,color='green',label='Global',alpha=0.1)

plt.title('Relation between Dislike and Error or Remove')
plt.legend(loc='upper right')
plt.xlabel('Removed Video')
plt.ylabel('Dislike')
plt.show()


# **What I understand is there is almost no interaction between them.**

# **I also wonder that is there any interaction between dislike and comment disabled.**

# In[ ]:


plt.scatter(dataFR.comments_disabled,dataFR.dislikes,color='blue',label='France',alpha=0.7)
plt.scatter(dataCA.comments_disabled,dataCA.dislikes,color='red',label='Canada',alpha=0.7)
plt.scatter(dataUS.comments_disabled,dataUS.dislikes,color='yellow',label='United States',alpha=0.7)
plt.scatter(dataIN.comments_disabled,dataIN.dislikes,color='purple',label='India',alpha=0.7)
plt.scatter(dataDE.comments_disabled,dataDE.dislikes,color='black',label='Germany',alpha=0.7)
plt.scatter(dataGB.comments_disabled,dataGB.dislikes,color='green',label='Global',alpha=0.1)

plt.title('Relation between Dislike and Comments Disabled')
plt.legend(loc='upper right')
plt.xlabel('Comments Disabled')
plt.ylabel('Dislike')
plt.show()


# **Similar with before**

# **Lastly, I want to see interaction between like and comment count.**

# In[ ]:


plt.scatter(dataFR.comment_count,dataFR.likes,color='blue',label='France',alpha=0.6)
plt.scatter(dataCA.comment_count,dataCA.likes,color='red',label='Canada',alpha=0.6)
plt.scatter(dataUS.comment_count,dataUS.likes,color='yellow',label='United States',alpha=0.6)
plt.scatter(dataIN.comment_count,dataIN.likes,color='purple',label='India',alpha=0.6)
plt.scatter(dataDE.comment_count,dataDE.likes,color='black',label='Germany',alpha=0.6)
plt.scatter(dataGB.comment_count,dataGB.likes,color='green',label='Global',alpha=0.1)

plt.title('Relation between Like and Comment')
plt.legend(loc='upper right')
plt.xlabel('Comment')
plt.ylabel('Like')
plt.show()


# **Looks like Canada and India don't like to comment and Germany is the most balanced.**
