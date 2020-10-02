#!/usr/bin/env python
# coding: utf-8

# 
# Word clouds are a handy tool "for quickly perceiving the most prominent terms and for locating a term alphabetically to determine its relative prominence."[1]
# 
# In this notebook, I compute the word clouds for all speakers during Mark Zuckerberg's Congressional Testimony.
# 
# 
# 
# *Mark Zuckerberg's Word Cloud*
# 
# ![Mark's Wordcloud](https://i.imgur.com/LDVtnNC.png)
# 
# [1] Wikipedia https://en.wikipedia.org/wiki/Tag_cloud
# 

# In[49]:


import pandas as pd

df = pd.read_csv('../input/mark.csv')
grp = df.groupby('Person')

speakers = grp['Text'].apply(' '.join).reset_index()
speakers = speakers[speakers['Person'].str.isupper()]  # CSV malformed
speakers['Text'] = speakers['Text'].str.replace('\r\n', ' ')
speakers['String Length'] = speakers['Text'].apply(len)

# The speakers dataframe contains all dialog spoken, grouped by each speaker
speakers


# In[68]:


import wordcloud
import matplotlib.pyplot as plt

def word_cloud(corpus, title):
    words = corpus.lower()
    cloud = wordcloud.WordCloud(background_color='black',
                                max_font_size=200,
                                width=1600,
                                height=800,
                                max_words=300,
                                relative_scaling=.5).generate(words)
    plt.figure(figsize=(16,10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(cloud)


# Lets look at all Senators combined, then we will break it down to each one.

# In[67]:


mask = df['Person'].str.contains('ZUCK')
senators = df[~mask]['Text'].str.replace('\r\n', ' ').values

word_cloud(' '.join(senators), 'All Senators')


# You might have to **double click the whitespace** next to the word cloud images for the notebook to expand all images

# In[56]:


for _, (title, text, _) in speakers.iterrows():
    word_cloud(text, title)

