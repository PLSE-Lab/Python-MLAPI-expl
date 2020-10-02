#!/usr/bin/env python
# coding: utf-8

# # Complete Analysis Of The Late Night Talk Show By Conan

# **Worked Really hard for this,Considering I am a Beginner,Would Really Make Me Happy If You Upvote It,If You Like It**
# 
# 
# 
# Thanking the Dataset Creator for this dataset

# **Late Night Talk Shows are a staple of American television culture and with the shows establishing a digital presence in the form of YouTube channels, this culture has become more global. Some of the channels here have more than 20 Million subscribers which shows the amount of influence they hold in this platform.**
# 
# 

# Importing necessary libraries.

# ![](http://www.google.com/urlsa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLatenight_talk_show&psig=AOvVaw1qrXPGiJ_AlgVXLsw_lqBP&ust=1594227281121000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPiakPbNu-oCFQAAAAAdAAAAABAE)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Conan's Hosting : Analysis

# Reading the file

# In[ ]:


conan = pd.read_csv("../input/late-night-talk-show-youtube-dataset/Conan.csv")
 #printing the first five rows 
conan.head()


# # Data Preprocessing & Cleaning

# First lets drop unwanted rows to reduce ambiguity

# In[ ]:


conan.drop(["publishedAtSQL","definition","videoCategoryId","caption","licensedContent"],axis='columns', inplace=True)
conan.head()


# Lets find the total number of rows present

# In[ ]:


conan.shape


# Finding and replacing the null values in the dataset

# In[ ]:


conan.isnull().sum()


# **Created a for loop and replaced the values with the mode(The most repeated values)**

# In[ ]:




for column in ['durationSec','viewCount','likeCount','dislikeCount','commentCount']:
    conan[column].fillna(conan[column].mode()[0], inplace=True)


# **For columns containing String values we find the index and remove it**

# In[ ]:


viddesnull = conan.loc[conan['videoDescription'].isnull()] 
  
print(viddesnull) 


# In[ ]:


vidcatnull = conan.loc[conan['videoCategoryLabel'].isnull()] 
  
print(vidcatnull) 


# In[ ]:


conanfinal=conan.drop([1475,2520,3678])
conanfinal.shape


# # Now there are no Null Values

# In[ ]:


conanfinal.isnull().sum()


# Here **we are finding the unique categories that exist under the video category column**

# In[ ]:


uniqconan=conanfinal.videoCategoryLabel.unique()
for values in uniqconan:
    print(values)


# **Finding the sum of total views in each category**

# In[ ]:


Comedy=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Comedy'].sum()
print(Comedy)


# In[ ]:


Ent=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Entertainment'].sum()
print(Ent)


# In[ ]:


Mus=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Music'].sum()
print(Mus)


# In[ ]:


Game=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Gaming'].sum()
print(Game)


# In[ ]:


Np=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'News & Politics'].sum()
print(Np)


# In[ ]:


TE=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Travel & Events'].sum()
print(TE)


# In[ ]:


PB=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'People & Blogs'].sum()
print(PB)


# In[ ]:


FA=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Film & Animation'].sum()
print(FA)


# In[ ]:


Ed=conanfinal.viewCount[conanfinal.videoCategoryLabel == 'Education'].sum()
print(Ed)


# # In-depth Exploratory Analysis

# In[ ]:


exp_vals=[Comedy,Ent,Mus]
labels=['Comedy','entertainment','Music']
plt.axis('equal')
explode=(0,0.5,0.5)
colors=['Pink','Red','Orange']
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=45,labels=labels,colors=colors)
plt.show()


# We can conclude that people prefer comedy and entertainment the most

# In[ ]:


exp_vals=[Game,Np,TE]
labels=['Game','News & Politics','Travel & Events']
plt.axis('equal')
explode=(0,0.5,0.5)
colors=['Orange','Brown','Yellow']
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=45,labels=labels,colors=colors)
plt.show()


# On further exploring the sub levels, the game enthralls more viewers

# In[ ]:


exp_vals=[PB,FA,Ed]
labels=['People & Blogs','Film & Animation','Education']
plt.axis('equal')
colors=['Pink','Red','Yellow']
explode=(0,0.5,0)
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=180,labels=labels,colors=colors,)
plt.show()


# Its funny that people prefer education & blogs the least of all the categories

# Thank You!
# !
# 
# [I am leaving a link of my linear regression model for another notebook! feel free to check it out! its simple]https://www.kaggle.com/sujay12345/prediction-of-you-getting-an-admit-in-us)
# 
# 
# # Your feedback is much appreciated
