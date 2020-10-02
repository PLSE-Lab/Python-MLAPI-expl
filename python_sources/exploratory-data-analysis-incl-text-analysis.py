#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (incl. text analysis)
# We will use this data set with 2017 instances and 16 attributes, among which there are 13 song features attributes and 3 attributes for "song_title", "artist" and "target". Based on the obtained data set, we are going to try some basic and interesting EDA. Here we go.

# **First we import some common libraries and generate the dataframe to analyse.**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("../input/data.csv")
df = df.drop("Unnamed: 0", axis="columns")
df.head()


# **For three categorical attributes - "key", "mode" and "time_signature", we create three countplot distributions to understand more about the audience's song preference.**

# In[ ]:


fig1 = plt.figure(figsize=(18, 12))

ax1 = fig1.add_subplot(331)
sns.countplot(x='key',hue='target',data=df, palette='BuGn')

ax2 = fig1.add_subplot(332)
sns.countplot(x='mode',hue='target',data=df, palette='BuGn')

ax3 = fig1.add_subplot(333)
sns.countplot(x='time_signature',hue='target',data=df, palette='BuGn')


# **Use pairplot to find out some possible correlated relationships between 13 attributes. **

# In[ ]:


sns.pairplot(df)


# **Use heapmap to review the correlation more clearly. **

# In[ ]:


fig2 = plt.figure(figsize=(16, 8))
sns.heatmap(df.corr(), annot=True, annot_kws={'weight':'bold'},linewidths=.5, cmap='YlGnBu')


# **"Loudness" and "energy" have highly positive correlated relationship (P=0.76). Use scatter chart and trend line to view their relationship by different targets. **

# In[ ]:


sns.lmplot(y='loudness',x='energy',data=df, hue='target',palette='BuGn')


# **"Energy" and "acousticness" have highly negative correlated relationship (P=-0.65). Use scatter chart and trend line to view their relationship by different targets. **

# In[ ]:


sns.lmplot(y='energy',x='acousticness',data=df, hue='target',palette='BuGn')


# **Generate wordclouds to get some interesting and frequent key words appearing in the "song_title" for target=1 and target=0. **

# In[ ]:


import wordcloud
from subprocess import check_output
get_ipython().run_line_magic('pylab', 'inline')

songs_like = ' '.join(df[df['target']==1]['song_title']).lower().replace(' ',' ')
cloud_like = wordcloud.WordCloud(background_color='white',
                            mask=imread('Spotify.jpg'),
                            max_font_size=100,
                            width=2000,
                            height=2000,
                            max_words=1000,
                            relative_scaling=.5).generate(songs_like)

songs_dislike = ' '.join(df[df['target']==0]['song_title']).lower().replace(' ',' ')
cloud_dislike = wordcloud.WordCloud(background_color='white',
                            mask=imread('Spotify.jpg'),
                            max_font_size=100,
                            width=2000,
                            height=2000,
                            max_words=1000,
                            relative_scaling=.5).generate(songs_dislike)
fig3=plt.figure(figsize=(12,12))

ax4 = fig3.add_subplot(121)
plt.imshow(cloud_like)
plt.axis('off')
plt.title('Like (target=1)', fontsize=20, color='g', fontweight='bold')

ax5 = fig3.add_subplot(122)
plt.imshow(cloud_dislike)
plt.axis('off')
plt.title('Dislike (target=0)', fontsize=20, color='g', fontweight='bold')


# You can download the 'Spotify.jpg' from Internet and save it into the same file as the Jupyter Notebook.
# Output as follows:
#  
# ![https://wx1.sinaimg.cn/mw690/97a025f4gy1g4i10uokfsj211g0iwnhb.jpg](http://)

# **Create a bar chart to get Top 10 Favorite Artists for the user. **

# In[ ]:


artist_like = df[df['target']==1].groupby('artist').count().reset_index()[['artist', 'target']]
artist_like.columns = ['artist', 'appearances']
artist_like = artist_like.sort_values('appearances', ascending=False)
artist_like=artist_like.head(10)
plt.barh(left=0, y='artist', width='appearances', data=artist_like, color='g', alpha=0.7)
plt.title('Top 10 Favorite Artists', color='g', fontsize='xx-large')


# Thanks for your attention. Actually, we, as a group, used this data set to finish a group project - a classification task, for a business intelligence course. So first thing first, thanks a lot for the creator of this data set. EDA was a small part of our project. More codes related to ML were preserved because they were not done by myself. It's my first time to create kernel. So happy to be here and make friends. Cheers~
