#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ### In this kernel, I will do some basic EDA on the data in this competition (except images) and see how the features are related with the target (the revenue).

# <center><img src="https://i.imgur.com/YGb66D5.png" width="300px"></center>

# ### Import necessary libraries
# 
# 

# In[ ]:


import os
import gc
import sys

import numpy as np
import pandas as pd

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# ### Load the training data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# In[ ]:


train_df.head()


# ### Extract the revenues from the dataframe

# In[ ]:


revenues = train_df['revenue']


# ## Budget
# ### Extract budgets from the data

# In[ ]:


budgets = train_df['budget']


# ### Visualize relationship between budget and revenue

# In[ ]:


sns.jointplot(x=budgets, y=revenues, dropna=True, color='blueviolet', kind='reg')
plt.show()


# There seems to be a positive correlation between budget and revenue. This implies that the revenue of a movie generally tends to increase when its budget increases. This is probably because the directors and producers can afford a better cast, a higher quality set, a more ambitious plot etc with a higher budget.

# ## Popularity
# ### Visualize the relationship between popularity and revenue

# In[ ]:


plot = sns.jointplot(x='popularity', y='revenue', data=train_df, dropna=True, color='orangered', kind='reg') 


# The popularity and revenue do not seem to have any real correlation. There is only a very slight positive correlation. This is probably because a more popular movie generates more revenue :)

# ## Language 
# ### Visualize the relationship between the original language and revenue of the *movie* 

# ### Language two-letter codes
# 
# Look [here](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) for the two-letter language codes.
# 

# ## Most profitable movie languages

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
ax.tick_params(axis='both', labelsize=12)
plt.title('Original Language and Revenue', fontsize=20)
plt.xlabel('Revenue', fontsize=16)
plt.ylabel('Original Language', fontsize=16)
sns.boxplot(ax=ax, x='revenue', y='original_language', data=train_df, showfliers=False, orient='h')
plt.show()


# Some languages seem to attract greater audiences than others and end up generating more revenue. **For example, the highest revenue movies are in English, Chinese and Turkish. ('en', 'zh' and 'tr'). Hindi ('hi') and Japanese ('ja') are not far behind.**
# 
# 
# 

# ## Most common languages

# In[ ]:


plt.figure(figsize = (12, 8))
text = ' '.join(train_df['original_language'])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Languages', fontsize=20)
plt.axis("off")
plt.show()


# The most common languages in the movie data seem to be English ('en'), French ('fr'), Russian ('ru'), Hindi ('hi') etc.

# ## Genre
# ### Visualize the relationship between the genre and revenue of the *movie* 

# In[ ]:


genres = []
repeated_revenues = []
for i in range(len(train_df)):
  if train_df['genres'][i] == train_df['genres'][i]:
      movie_genre = [genre['name'] for genre in eval(train_df['genres'][i])]
      genres.extend(movie_genre)
      repeated_revenues.extend([train_df['revenue'][i]]*len(movie_genre))
  
genre_df = pd.DataFrame(np.zeros((len(genres), 2)))
genre_df.columns = ['genre', 'revenue']
genre_df['genre'] = genres
genre_df['revenue'] = repeated_revenues


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
ax.tick_params(axis='both', labelsize=12)
plt.title('Genres and Revenue', fontsize=20)
plt.xlabel('revenue', fontsize=16)
plt.ylabel('genre', fontsize=16)
sns.boxplot(ax=ax, x=repeated_revenues, y=genres, showfliers=False, orient='h')
plt.show()


# It looks like some movie genres tend to earn more revenue than others on average. Animation and Adventure movies lead the way in terms of revenue, but Family and Fantasy are not far behind.

# ## Most common movie genres

# In[ ]:


plt.figure(figsize = (12, 8))
text = ' '.join(genres)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=2000, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Genres', fontsize=30)
plt.axis("off")
plt.show()


# The most common movie genres seem to be Drama, Comedy and Thriller.

# ## Tagline Sentiment
# ### Analyze sentiment of movie taglines

# In[ ]:


def sentiment(x):
  if type(x) == str:
    return SIA.polarity_scores(x)
  else:
    return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}


# ### Create NLTK Vader Sentiment Analyzer
# It measures the positivity, neutrality, negativity and compoundness of the taglines

# In[ ]:


SIA = SentimentIntensityAnalyzer()
overview_sentiments = train_df['overview'].apply(lambda x: sentiment(x))
tagline_sentiments = train_df['tagline'].apply(lambda x: sentiment(x))


# ### Calculate sentiments

# In[ ]:


neutralities = [sentiment['neu'] for sentiment in tagline_sentiments]
negativities = [sentiment['neg'] for sentiment in overview_sentiments]
compound = [sentiment['compound'] for sentiment in overview_sentiments]


# ## Negativity of overview

# ### Visualize relationship between negativity of overview and revenue
# 
# 
# 
# 

# In[ ]:


sns.jointplot(x=negativities, y=revenues, dropna=True, color='mediumvioletred', kind='scatter')
plt.show()


# There seems to be a negative correlation between negativity of the overview and revenue. This is probably because the more negative the tagline, the less likely people are to watch the movie. This is because the negative tagline makes them not want to watch the movie. **This is probably why there is a massive peak in revenue at the lowest negativities close to 0.**
# 
# *   List item
# *   List item
# 
# 

# ## Neutrality of tagline

# ### Visualize relationship between neutrality of tagline and revenue of movies

# In[ ]:


sns.jointplot(x=neutralities, y=revenues, dropna=True, color='mediumblue', kind='reg')
plt.show()


# The neutrality of the tagline and revenue of the movie seem to be positively correlated. **This is probably because a movie with a more inclusive movie tagline which does not denounce any political or religious ideology is more likely to have a larger audience.** Thisis why the line of best fit has a positive slope. This is probably why there is large peak at the maximum neutrality point (1.0).

# ## Compoundness of overview
# ### Visualize relationship between compoundness of tagline and revenue of movies

# In[ ]:


sns.jointplot(x=compound, y=revenues, dropna=True, color='maroon', kind='reg')
plt.show()


# There does not seem to be any apparent relationship between the compoundness (grammatical complexity) of the overview and its revenue.

# ## Overview lengths

# In[ ]:


lengths = train_df['tagline'].apply(lambda x: len(str(x)))


# In[ ]:


sns.jointplot(x=lengths, y=revenues, dropna=True, color='crimson')


# There seems to be a clear positive correlation between the length of the overview and the revenue. This is probably because short and catchy taglines are more likely to attact more audience than long, explanatory ones.

# ### That's it ! Thanks for reading my kernel ! Hope you found it useful :)

# ### Please post your feedback and suggestions in the comments below.
