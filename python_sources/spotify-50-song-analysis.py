#!/usr/bin/env python
# coding: utf-8

# # INFORMATION 
# Spotify Technology S.A. is an international media services provider of Swedish origin. It is legally domiciled in Luxembourg and is headquartered in Stockholm, Sweden. Founded in 2006, the company's primary business is providing an audio streaming platform, the "Spotify" platform, that provides DRM-protected music, videos and podcasts from record labels and media companies. As a freemium service, basic features are free with advertisements or automatic music videos, while additional features, such as offline listening and commercial-free listening, are offered via paid subscriptions.
# 
# Launched on October 7, 2008, the Spotify platform provides access to over 50 million tracks. Users can browse by parameters such as artist, album, or genre, and can create, edit, and share playlists. Spotify is available in most of Europe and the Americas, Australia, New Zealand, and parts of Africa and Asia, and on most modern devices, including Windows, macOS, and Linux computers, and iOS, and Android smartphones and tablets. As of October 2019, the company had 248 million monthly active users, including 113 million paying subscribers.
# 
# Unlike physical or download sales, which pay artists a fixed price per song or album sold, Spotify pays royalties based on the number of artists streams as a proportion of total songs streamed. It distributes approximately 70% of its total revenue to rights holders, who then pay artists based on their individual agreements. Spotify has faced criticism from artists and producers including Taylor Swift and Thom Yorke, who have argued that it does not fairly compensate musicians. In 2017, as part of its efforts to renegotiate license deals for an interest in going public, Spotify announced that artists would be able to make albums temporarily exclusive to paid subscribers if the albums are part of Universal Music Group or the Merlin Network.
# 
# Spotify's international headquarters are in Stockholm, Sweden, though each region has its own headquarters. Since February 2018, it has been listed on the New York Stock Exchange and in September 2018, the company relocated its New York City offices to 4 World Trade Center.

# It is taken from wikipedia. -->https://en.wikipedia.org/wiki/Spotify

# <img src="https://i.internethaber.com/storage/files/images/2019/11/01/spofity-yeni-yBmg_cover.jpg" style="width:1000px;height:500px;">

# # LIBRARIES

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore",category = DeprecationWarning)
warnings.filterwarnings("ignore",category = FutureWarning)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **We can make import as below** 

# In[ ]:


filename='/kaggle/input/top50spotify2019/top50.csv'
spoti=pd.read_csv(filename,encoding='ISO-8859-1')
spoti.head(10) # It shows first 10 row information in data 


# In[ ]:


spoti.info()
#It gives information about data.


# In[ ]:


# Show some statistics about dataset
spoti.describe()


# In[ ]:


spoti.columns
#It gives what we have columns


# In[ ]:


gb_genre =spoti.groupby("Genre").sum()
#It classified by genre variable 
gb_genre.head()


# In[ ]:


#Calculates the number of rows and columns
print(spoti.shape)


# # Visualization of Data

# In[ ]:


#Heatmap
plt.figure(figsize=(10,10))
plt.title('Correlation Map')
ax=sns.heatmap(spoti.corr(),
               linewidth=3.1,
               annot=True,
               center=1)


# In[ ]:


#Boxplot
#It shows outlier values and value of popularity
sns.boxplot( y = spoti["Popularity"])
plt.show()


# In[ ]:


#Catplot
#It gives count of genre in spotify top 50 list. 
sns.catplot(y = "Genre", kind = "count",
            palette = "pastel", edgecolor = ".6",
            data = spoti)
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.jointplot(x=spoti["Beats.Per.Minute"].values, y=spoti['Popularity'].values, size=10, kind="kde",)
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Beats.Per.Minute", fontsize=12)
plt.title("Beats.Per.Minute Vs Popularity", fontsize=15);
#The purpose of this graph is to show connection among Beats and Popularity


# In[ ]:


threshold = sum(spoti.Energy)/len(spoti.Energy)
print(threshold)
spoti["Energy_level"] = ["energized" if i > threshold else "without energy" for i in spoti.Energy]
spoti.loc[:10,["Energy_level","Energy"]]
#This caught my attention to the effect of energy level on music in here and i calcuted it. It classified according to mean of value


# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(x='Loudness..dB..', y='Popularity', data=spoti)
plt.xlabel('Loudness..dB..', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
plt.show()
# I want to show relationship loudness and popularity. From there we can learn to contribution of loudness level to popularity


# In[ ]:


# Some kind of Histogram Plot
f, ax = plt.subplots(figsize=(10,8))
x = spoti['Loudness..dB..']
ax = sns.distplot(x, bins=10)
plt.show()


# In[ ]:


sns.pairplot(spoti)
plt.plot()
plt.show()
# It shows all histogram graph with data colums.


# In[ ]:


sns.lmplot(x="Energy",y="Popularity",data=spoti,size=10,hue="Genre")

plt.plot()
plt.show()
#this graph is so attractive because of different from other. My target in there is to show to Excellence of connection of Energy and Poularity  


# **please upvote if you liked it.Thank you**
