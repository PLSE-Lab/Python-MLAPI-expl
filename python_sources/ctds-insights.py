#!/usr/bin/env python
# coding: utf-8

# # Chai Time Data Science Insights

# ## 1. About

# Chai Time Data Science is a Podcast + Video + Blog based show for interviews with practitioners, researchers and kagglers and all things data science. It is also a continuation of 'Interview with Machine Learning Heroes Series' by Sanyam Bhutani. The poddcasts are available on YouTube, Spotify, Apple Music as well as on all other major podcast directories. 
# In case you want to learn more about Chai Time Data Science, here's the link for your reference: https://sanyambhutani.com/tag/chaitimedatascience/

# ## 2. Analysis

# Let us start by importing the required libraries.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly_express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import display, HTML
import requests
from bs4 import BeautifulSoup
import re
import json

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


subtitle_path = '/kaggle/input/chai-time-data-science/Cleaned Subtitles/'
episodes_df = pd.read_csv('/kaggle/input/chai-time-data-science/Episodes.csv')
PLOT_BGCOLOR='#DADEE3'
PAPER_BGCOLOR='rgb(255,255,255)'

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# Let us take a look at the Episodes data.

# In[ ]:


# Visualizing the top 5 rows of the episodes data frame
episodes_df.head(5)


# In[ ]:


# Checking the shape of the data
episodes_df.shape


# So we can see that there are 85 episodes of CTDS show available.

# Let us see if there are any missing values in the Episodes data set or not.

# In[ ]:


# Checking for missing values
episodes_df.isnull().sum()


# We can see here that the attributes 'heroes', 'heroes_gender', 'heroes_location', 'heroes_nationality', 'heroes_kaggle_username' and 'heroes_twitter_handle'have the most missing values. We will drop the missing values as they can skew the data. 

# Now we are going to drop the missing values, and again take a look at the data set to see whether the missing values have been removed or not.

# In[ ]:


# Dropping the missing values
episodes_df.dropna()


# # Gender Distribution on CTDS

# In[ ]:


# Let us now look at the gender distribution on the show
import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(x = "heroes_gender", data = episodes_df)


# Clearly, we can see that there are very few females on CTDS.

# # Location of Heroes

# In[ ]:


# Let us now take a look at the different locations of the heroes
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y = "heroes_location", data = episodes_df)


# We can see that most number of Heroes are either located in USA or Canada.

# # Category of Heroes

# In[ ]:


# Let us now look at the category of the heroes
sns.set(style="darkgrid")
ax = sns.countplot(x = "category", data = episodes_df)


# The people interviewed on CTDS are mostly from the Industry or from Kaggle, people in research are not interviewed as much as other categories.

# # Preference of Tea flavour

# In[ ]:


# Let us take a look at the different flavours of tea preferred on the show
f, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y = "flavour_of_tea", data = episodes_df)


# We can see here that Masala Chai and Ginger Chai is the most preferred by the Heroes on the show while Kashmiri Kahwa is preferred by only a few.

# # Recording times of the show

# In[ ]:


# Let us now take a look at the recording times
f, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y = "recording_time", data = episodes_df)


# Most of the shows have been recorded at night.

# # Episode Duration in Seconds

# In[ ]:


# Let us now take a look at the episode duration
episodes_df['episode_duration'].plot(x = 'Episodes', y = 'Duration in seconds')


# The highest episode duration is approximately 8000 seconds, while the lowest episode duration is of a few 100 seconds.

# # Statistics for different platforms (YouTube, Spotify, Apple)

# In[ ]:


# Let us take a look at some video statistics on YouTube


total_views = episodes_df['youtube_views'].sum()
total_impressions = episodes_df['youtube_impressions'].sum()
total_likes = episodes_df['youtube_likes'].sum()
total_dislikes = episodes_df['youtube_dislikes'].sum()
total_comments = episodes_df['youtube_comments'].sum()
total_subscribers = episodes_df['youtube_subscribers'].sum()

print("Total number of views on Youtube:", total_views)
print("Total number of youtube impressions:", total_impressions)
print("Total number of likes on youtube:", total_likes)
print("Total number of dislikes:", total_dislikes)
print("Total number of comments on youtube:", total_comments)
print("Total number of youtube subscribers:", total_subscribers)


# In[ ]:


# Let us now take a look at the spotify statistics
total_streams = episodes_df['spotify_streams'].sum()
total_listeners = episodes_df['spotify_listeners'].sum()

print("Total number of spotify streams:", total_streams)
print("Total number of listeners on spotify:", total_listeners)


# In[ ]:


# Let us now take a look at the apple statistics
total_listeners_a = episodes_df['apple_listeners'].sum()

print("Total number of listeners on apple:", total_listeners_a)


# # Comparing the Platforms

# In[ ]:


# Creating a list
data = {'Name of the Platform':['YouTube', 'Spotify', 'Apple'], 'Total number of Views/Listeners':[43616, 5455, 1714]} 
  
# Create=ing the dataFrame 
df1 = pd.DataFrame(data) 
  
print(df1) 


# We can see here that YouTube has more number of views while Spotify has more listeners as compared to Apple. So we can say that YouTube is the most preferred platform to view the videos of CTDS while Spotify is preferred over Apple for listening to Podcasts of CTDS.

# # Most viewed episode in the Kaggle category

# In[ ]:


# Let us now take a look at the most viewed episodes in the 'Kaggle' category
most_viewed = episodes_df[episodes_df['category'] == 'Kaggle'].sort_values(by = 'youtube_views',ascending = False)
#Top 3 most viewed shows in Kaggle category
most_viewed.head(3)


# # Most viewed episode in the 'Industry' category

# In[ ]:


# Let us now take a look at the most viewed episodes in the category 'Industry'
most_viewed2 = episodes_df[episodes_df['category'] == 'Industry'].sort_values(by = 'youtube_views',ascending = False)
#Top 3 most viewed shows in Kaggle category
most_viewed2.head(3)


# # Most liked episode on YouTube

# In[ ]:


# Most liked episode on Youtube
most_liked = episodes_df.sort_values(by = 'youtube_likes', ascending = False)
most_liked.head(3)
# Top 3 most liked videos


# # Episode with most subscribers on YouTube

# In[ ]:


# Most subcribers on the episodes
most_subscribers = episodes_df.sort_values(by = 'youtube_subscribers', ascending = False)
most_subscribers.head(3)
# Top 3 most subscribed episodes


# **From the above analysis, we can say that the 27th episode is the most viewed (in the 'Industry' category), liked episode of Chai Time Data Science, as well has the most subscribers on YouTube.** 

# In the 27th episode, Sanyam Bhutani has interviewed [Jeremy Howard](https://www.kaggle.com/jhoward), who is an entrepreneur, business strategist, developer and an educator. He is a founding researcher at fast.ai which is a research institue dedicated to make deep learning more accessible.
# 

# # Let us now dive in to the 27th Episode to find out more details.

# In[ ]:


# Reading the cleaned subtitles of the 27th episode file
cleaned_subtitles = pd.read_csv('/kaggle/input/chai-time-data-science/Cleaned Subtitles/E27.csv')


# # Visualizing the subtitles

# In[ ]:


# Let us now take a look at the subtitles file
cleaned_subtitles.head(5)


# # Most used words in conversation in the 27th Epiode

# In[ ]:


# Let us take a look at the words which dominate the conversation
from wordcloud import WordCloud, ImageColorGenerator


word_cloud = WordCloud(width = 1000,
                       height = 800,
                       colormap = 'Blues', 
                       margin = 0,
                       max_words = 200,  
                       min_word_length = 4,
                       max_font_size = 120, min_font_size = 15,  
                       background_color = "white").generate(" ".join(cleaned_subtitles['Text']))

plt.figure(figsize = (10, 15))
plt.imshow(word_cloud, interpolation = "gaussian")
plt.axis("off")
plt.show()


# In[ ]:


# Defining a function to visualise n-grams
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]


# # Most used bigrams in the conversation

# In[ ]:


# Visualising the most frequent bigrams occurring in the conversation
from sklearn.feature_extraction.text import CountVectorizer
top_bigrams = get_top_ngram(cleaned_subtitles['Text'],2)[:10]
x,y = map(list,zip(*top_bigrams))
sns.barplot(x = y,y = x)


# # Most used trigrams in the conversation

# In[ ]:


# Visualising the most frequent trigrams occurring in the conversation
from sklearn.feature_extraction.text import CountVectorizer
top_trigrams = get_top_ngram(cleaned_subtitles['Text'],3)[:10]
x,y = map(list,zip(*top_trigrams))
sns.barplot(x = y,y = x)


# # Sentiment Analysis of Episode 27

# In[ ]:


# TextBlob library provides a consistent API for NLP tasks such as POS Tagging, noun-phrase extraction and sentiment analysis
from textblob import TextBlob
TextBlob('100 people died yesterday due to COVID-19').sentiment


# Let us look at the polarity score of the statements used in the conversation during Episode 27.

# In[ ]:


# Defining a function to check the sentiment polarity (whether it is positive or negative)
def polarity(text):
    return TextBlob(text).sentiment.polarity

cleaned_subtitles['polarity_score'] = cleaned_subtitles['Text'].   apply(lambda x : polarity(x))
cleaned_subtitles['polarity_score'].hist()


# Visualizing total number of positive, negative and neutral statements used in the conversation.

# In[ ]:


# Defining a function to classify the sentiment based on the polarity 
def sentiment(x):
    if x<0:
        return 'neg'
    elif x==0:
        return 'neu'
    else:
        return 'pos'
    
cleaned_subtitles['polarity'] = cleaned_subtitles['polarity_score'].   map(lambda x: sentiment(x))

plt.bar(cleaned_subtitles.polarity.value_counts().index,
        cleaned_subtitles.polarity.value_counts())


# # Text having positive sentiment

# In[ ]:


# Printing text having a positive sentiment
cleaned_subtitles[cleaned_subtitles['polarity'] == 'pos']['Text'].head()


# # Text having negative sentiment

# In[ ]:


# Printing text having a negative sentiment
cleaned_subtitles[cleaned_subtitles['polarity'] == 'neg']['Text'].head()


# # Text having neutral sentiment

# In[ ]:


# Printing text having a neutral sentiment
cleaned_subtitles[cleaned_subtitles['polarity'] == 'neu']['Text'].head()

