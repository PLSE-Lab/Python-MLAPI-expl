#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS


# # Loading Data
# 

# In[ ]:


data = pd.read_csv('../input/googleplaystore.csv')
data.head()


# In[ ]:


data.shape


# So we are studying 10841 Google applications in this dataset. Their average rating is:

# In[ ]:


data["Rating"].mean()


# It seems not so bad! Android users are quite happy...

# # Dataset Cleaning

# In[ ]:


data["Installs"] = data["Installs"].map(lambda x: x.rstrip('+'))
data["Size"] = data["Size"].map(lambda x: x.rstrip('M'))


# In[ ]:


data = data[data.Installs != 'Free']
data["Installs"] = data["Installs"].map(lambda x:  int(x.replace(',' , '')))


# In[ ]:


data['Reviews']=data['Reviews'].astype('int')
data['Installs']=data['Installs'].astype('int')
data['Rating']=data['Rating'].astype('float64')


# # Plots

# In[ ]:


type_counts = data.groupby(['Type']).count()
plt.pie(type_counts["App"], labels = data.groupby(['Type']).groups.keys())
plt.show()


# In[ ]:


category_counts = data.groupby(['Category']).count()
labels = data.groupby(['Category']).groups.keys()
patches, texts = plt.pie(category_counts["App"])
plt.legend(patches, labels, loc='center right', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
plt.show()


# In[ ]:


install_counts = data.groupby(['Installs']).count()
labels = data.groupby(['Installs']).groups.keys()
patches, texts = plt.pie(install_counts["App"] )
plt.legend(patches, labels, loc='center right', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
plt.show()


# Is there any correlation between installs and reviews? It seems there should be some...

# In[ ]:


plt.scatter(data["Installs"], data["Reviews"])
plt.xlabel("Installs")
plt.ylabel("Reviews")
plt.xscale('log')
plt.yscale('log')
plt.xlim(1)
plt.ylim(1)
plt.show()


# Is there any correlation between installs and rating?

# In[ ]:


plt.scatter(data["Installs"], data["Rating"])
plt.xscale('log')
plt.xlabel("Installs")
plt.ylabel("Rating")
plt.show()


# How can we define the most popular app? Not sure this is correct but let's calculate what app has as many reviews as possible in our dataset. So the winner is:

# In[ ]:


data.loc[data['Reviews'].idxmax()]


# So Facebook  has got  78158306 reviews. Have you heard about this app? :)

# # User Reviews

# In[ ]:


reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
reviews.head()


# In[ ]:


reviews.shape


# There are 64295 reviews in our dataset.

# In[ ]:


reviews.groupby(['Sentiment']).size()


# In[ ]:


sentiments = reviews.groupby(['Sentiment']).count()
plt.pie(sentiments["App"], labels = sentiments.groupby(['Sentiment']).groups.keys())
plt.show()


# # Most popular Words

# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(reviews['Translated_Review'].dropna()))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(reviews['App'].dropna()))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# To be continued... I would like to investigate more how different features behave differently in various categories, for what apps Android users are ready to pay, etc. If you have any ideas what else to investigate about Google apps, please tell me in comments :)
