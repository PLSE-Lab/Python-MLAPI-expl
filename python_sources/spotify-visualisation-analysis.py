#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS


# In[ ]:


data=pd.read_csv("../input/top50spotify2019/top50.csv",encoding="ISO-8859-1")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


data.describe().T


# In[ ]:


corr=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)


# In[ ]:


#renaming the cols to convenience 
data.rename(columns = { "Unnamed: 0" : "id",
                        "Acousticness.." : "Acousticness",
                        "Track.Name" : "Track_Name" ,
                        "Valence." : "Valence",
                        "Length." : "Length",
                        "Loudness..dB.." : "Loudness_dB" ,
                        "Artist.Name" : "Artist_Name",
                        "Beats.Per.Minute" :"Beats_Per_Minute",
                        "Speechiness." : "Speechiness"},inplace = True)


# In[ ]:


#types of Genres and count
from collections import Counter
Counter(data['Genre'])


# In[ ]:


sns.distplot(data["Popularity"])


# In[ ]:


sns.scatterplot(x=data.Liveness,y=data.Popularity,data=data)


# In[ ]:


sns.scatterplot(x=data.Danceability,y=data.Popularity,data=data)


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(data.Genre)


# In[ ]:


#!pip install squarify


# In[ ]:


import squarify
plt.figure(figsize=(14,8))
squarify.plot(sizes=data.Artist_Name.value_counts(), label=data["Artist_Name"], alpha=.8 )
plt.axis('off')
plt.show()


# In[ ]:


#artist popularity in the data
data.groupby('Artist_Name').size().plot.bar()


# In[ ]:


#WordCloud


# In[ ]:


#top Genres listened.
string=str(data.Genre)
plt.figure(figsize=(12,8))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()

