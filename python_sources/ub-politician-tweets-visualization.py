#!/usr/bin/env python
# coding: utf-8

# # Visualization
# 
# In this notebook we will dive inside the dataset and perform some fancy visualization operations to get a nice overview of the data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


# First of all we need to load the datasets
train_df = pd.read_excel('../input/train.xlsx')
test_df = pd.read_excel('../input/test.xlsx')


# # Class weighthings
# The first important thing that we may want to notice is if we are facing a balanced or unbalanced dataset. To do so we will count the occurences of each political party contained in the dataset.

# In[ ]:


# Count the number of tweets for each political party
plt.figure()
train_count = train_df['party'].value_counts()
sns.barplot(x=train_count.index, y=train_count.values)
plt.figure()

# Count the number of tweets for each user
username_count = train_df['username'].value_counts()
sns.barplot(x=username_count.index, y=username_count.values)
_ = plt.xticks(rotation=90)


# # Word Cloud Visualization
# A useful tool to overview a problem like this is using wordcloud.
# 
# The following code block includes some useful actions to perform over data.

# In[ ]:


import re
from wordcloud import WordCloud
from stop_words import get_stop_words

def get_full(username, df):
    """ Get all tweets from a user
    """
    text_series = df[df['username'] == username]['text']
    all_text = " ".join(text_series)
    return all_text

def stop_words():
    """ Get the stop words
    """
    return get_stop_words('es') + get_stop_words('ca') + get_stop_words('en') + ['dels', 'ho', 'hi', 'pel'] + ['aren', 'can', 'couldn', 'des', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']

def filter_mentions(text):
    """ Filter all mentions
    """
    return re.sub("@\S+", "", text)

def filter_hashtags(text):
    """ Filter all hashtags
    """
    return re.sub("#\S+", "", text)

def get_hashtags_only(text):
    """
    """
    return re.findall("#\S+", text)


# In[ ]:


# Define the needed functions for wordcloud
def draw_word_cloud(username, df):
    text = filter_hashtags(filter_mentions(get_full(username, df)))
    wordcloud = WordCloud(
        stopwords=stop_words(),
        normalize_plurals=False,
        background_color='white').generate(get_full(username, df))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# ... And for cloud using only hashtags
def draw_hashtag_word_cloud(username, df):
    """ Draw the word cloud using just hashtags
    """
    text = get_hashtags_only(filter_mentions(get_full(username, df)))
    wordcloud = WordCloud(
        stopwords=stop_words(),
        normalize_plurals=False,
        background_color='white').generate(' '.join(text))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[ ]:


plt.title("Ines Arrimadas (C's)")
draw_word_cloud('inesarrimadas', train_df)
plt.title("Carles Puigdemont (JXCAT)")
draw_word_cloud('krls', train_df)
plt.title("Miquel Iceta (PSC)")
draw_word_cloud('miqueliceta', train_df)
plt.title("Xavier Domenech (Podem)")
draw_word_cloud('xavierdomenechs', train_df)
plt.title("Xavier Garcia Albiol (PPC)")
draw_word_cloud('albiol_xg', train_df)
plt.title("Marta Rovira (ERC)")
draw_word_cloud('martarovira', train_df)
plt.title("Albert Rivera (C's)")
draw_word_cloud('albert_rivera', train_df)


# In[ ]:


plt.title("Ines Arrimadas (C's)")
draw_hashtag_word_cloud('inesarrimadas', train_df)
plt.title("Albert Rivera (C's)")
draw_hashtag_word_cloud('albert_rivera', train_df)
plt.title("Carles Puigdemont (JXCAT)")
draw_hashtag_word_cloud('krls', train_df)
plt.title("Miquel Iceta (PSC)")
draw_hashtag_word_cloud('miqueliceta', train_df)
plt.title("Xavier Domenech (Podem)")
draw_hashtag_word_cloud('xavierdomenechs', train_df)
plt.title("Xavier Garcia Albiol (PPC)")
draw_hashtag_word_cloud('albiol_xg', train_df)
plt.title("Marta Rovira (ERC)")
draw_hashtag_word_cloud('martarovira', train_df)


# In[ ]:




