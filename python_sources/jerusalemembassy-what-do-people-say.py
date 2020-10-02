#!/usr/bin/env python
# coding: utf-8

# # #JerusalemEmbassy: What do people say?
# In this kernel, we are exploring this interesting dataset of #JerusalemEmbassy tweets.
# Twitter is always a very good sample of peoples' opinion on very hot topics like this recent Trump's decision. Let's see what insights the data has for us.. 
# 
# ### Loading
# First, we load our data and show our first five rows.

# In[ ]:


import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import wordcloud

import matplotlib.pyplot as plt
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
df = pd.read_csv('../input/Jerus20k.csv', encoding='ISO-8859-1')
df.head(5)


# Let's get a feeling of the data by inspecting different characteristics.
# 

# In[ ]:


print('Number of rows: {}'.format(df.shape[0]))
print('Number of columns: {}'.format(df.shape[1]))
print('Data description')
print(df.describe())


# ### What's the most retweeted tweet?

# In[ ]:


df = df.sort_values('retweetCount', ascending=False)
text = df.iloc[0]['text']

print('Most retweeted tweet with {} retweets:' .format(max(df['retweetCount'])))
print(text)


# ### What's the most favorited tweet?
# 

# In[ ]:


df = df.sort_values('favoriteCount', ascending=False)
text = df.iloc[0]['text']
print('Most favorited tweet with {} favorites:' .format(max(df['favoriteCount'])))
print(text)


# ### Which accounts tweeted the most? 

# In[ ]:


values = df['screenName'].value_counts()
values.head(5)


# ### Frequency of tweets over time
# 
# We aim to see how the frequency of the tweets vary over time. As expected, the day after Trump's decision announced, the tweets are much more than the following days, where they gradually decrease.

# In[ ]:


data = [go.Histogram(x=df['created'])]
py.iplot(data)


# ##  Sentiment Analysis of the tweets
# 
# It is interesting to see what are the sentiments of people tweeting about the subject.
# 
# We use VADER a sentiment analysis library based on lexicons of sentiment-related words. For each one of the tweet texts, we find the polarity scores positive, negative, neutral and compound. Eventually, we choose to use compound which is a normalized score (-1, 1) that takes into account the individual sentiment scores of the words.
# 
# We apply it and we check the first rows of the resulted dataframe.
# 

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

df['polarities'] = df['text'].apply(sid.polarity_scores)
df[['compound', 'neg', 'neu', 'pos']] = df['polarities'].apply(pd.Series)
df.head(5)


# The closer the compound score to 1, the more positive is the sentiment of the text, and the closer to -1, the more negative.
# 
# Let's check how many tweets are having score >=0 and are closer to positive sentiment and how many <0 and are closer to negative one.

# In[ ]:


print('Number of tweets with positive sentiment:', len(df.loc[df['compound']>=0]))
print('Number of tweets with negative sentiment:', len(df.loc[df['compound']<0]))


# ##  Wordclouds
# 
# * Wordcloud using the whole amount of tweets

# In[ ]:


from nltk.corpus import stopwords
stopwords = stopwords.words("english")

def clean_data(col):
    """Removes @mentions, <tags>, stopwords, urls, RTs, applies lower()"""
    allwords = ' '.join(col)
    tags_pattern = re.compile(r"<.*?>|(@[A-Za-z0-9_]+)")
    allwords = tags_pattern.sub('', allwords)
    allwords = re.sub(r'https\S+', '', allwords, flags=re.MULTILINE)
    allwords = allwords.replace('RT ', '').lower()
    allwords = ' '.join([word for word in allwords.split() if word not in stopwords])

    return allwords


# In[ ]:


allwords = clean_data(df.text)
cloud = wordcloud.WordCloud(background_color='white',
                            colormap='Blues',
                            max_font_size=200,
                            width=1000,
                            height=500,
                            max_words=300,
                            relative_scaling=0.5,
                            collocations=False).generate(allwords)
plt.figure(figsize=(20,15))
plt.imshow(cloud, interpolation="bilinear")


# *  Wordcloud using the positive sentiment tweets 

# In[ ]:


positive_df = df.loc[df['compound']>=0]
allwords = clean_data(positive_df.text)
cloud = wordcloud.WordCloud(background_color='white',
                            colormap='Greens',
                            max_font_size=200,
                            width=1000,
                            height=500,
                            max_words=300,
                            relative_scaling=0.5,
                            collocations=True).generate(allwords)
plt.figure(figsize=(20,15))
plt.imshow(cloud, interpolation="bilinear")


# * Wordcloud using negative sentiment words

# In[ ]:


negative_df = df.loc[df['compound']<0]
allwords = clean_data(negative_df.text)
cloud = wordcloud.WordCloud(background_color='white',
                            colormap='Reds',
                            max_font_size=200,
                            width=1000,
                            height=500,
                            max_words=300,
                            relative_scaling=0.5,
                            collocations=True).generate(allwords)
plt.figure(figsize=(20,15))
plt.imshow(cloud, interpolation="bilinear")

