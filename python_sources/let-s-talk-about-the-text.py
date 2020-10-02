#!/usr/bin/env python
# coding: utf-8

# ## What can we tell from the text of a Pitchfork review? Is the length important? Or the lexical density perhaps? Do these vary by genre? Or author? Or the year in which the album was released?

# ### All of these questions, and more, will be answered in the Kernel that follows!! Not to mention more handy practice for grouping dataframes in pandas, and matplotlib visualizations! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


con = sqlite3.connect('../input/database.sqlite')

artists = pd.read_sql('SELECT * FROM artists', con)
content = pd.read_sql('SELECT * FROM content', con)
genres = pd.read_sql('SELECT * FROM genres', con)
labels = pd.read_sql('SELECT * FROM labels', con)
years = pd.read_sql('SELECT * FROM years', con)
reviews = pd.read_sql('SELECT * FROM reviews', con)

con.close()


# In[ ]:


df1 = pd.merge(genres,reviews,on='reviewid')
df2 = pd.merge(labels,df1,on='reviewid')
df3 = pd.merge(content,df2,on='reviewid')
df_final = pd.merge(years,df3,on='reviewid')
df_final.head()


# ### Let's just preserve the columns that will be immediately relevant to our analysis. We drop all the row entries with NaNs for now. 

# In[ ]:


df_final.drop(['url','author_type','pub_date','pub_weekday','pub_day','pub_month'],axis=1,inplace=True)
df_final.dropna(inplace=True)
df_final['year'] = df_final['year'].astype(int)
df_final['pub_year'] = df_final['pub_year'].astype(int)
df_final.head()


# In[ ]:


df_final[df_final.duplicated('reviewid', keep=False)].sort_values('reviewid').head(5)


# ### We see that some of the reviewid's are duplicated since a band is listed as different genres, e.g. Abilene's self-titled album is listed as Metal and Rock in 2 separate rows. A little research suggests that Abilene was indeed Rock, but how do we generalize the drop_duplicates to only preserve the most likely genre?

# ### Best way would be to drop the duplicate which occurred least in the artist's list of genres, but there could be further ambiguity if one band had equal numbers of albums listed under separate genres. Skipping this for now, and dropping duplicates in reviewID, preserving the 1st. 

# In[ ]:


df_final.drop_duplicates('reviewid',inplace=True)
df_final.shape


# ### Let's compute some basic metrics on the review texts. Highly recommend checking out https://www.kaggle.com/pratapvardhan/kanye-lyrics-eda-song-generator-topic-modelling for most of the logic behind what I am about to do. 

# In[ ]:


df_final['text_len'] = df_final['content'].apply(lambda x: len(x))
df_final['text_words'] = df_final['content'].apply(lambda x: len(x.split()))
df_final['uniq_words'] = df_final['content'].apply(lambda x: len(set(x.split())))
df_final['lex_den'] = 100*df_final['uniq_words']/df_final['text_words']
df_final.head()


# ### Note: There don't seem to be any line breaks in the text content of the reviews, so we can't calculate the number of sentences per review

# In[ ]:


df_final[['text_len','text_words','uniq_words','lex_den']].hist(sharey=True, layout=(2, 2), figsize=(14, 12), color='b', alpha=.75, grid=False,bins=50);


# ### Lexical Density seems to be a pretty normal distribution, while the other parameters tend to be somewhat skewed, though these metrics would be more interesting if we were analyzing song lyrics or poetry instead of music reviews, where unique words and total words might give insights into an artist or an album.
# ### However let's look at these distributions by the top 10 authors.

# In[ ]:


df_final.author.value_counts()[:10].plot(kind='barh', figsize=(8,6));
plt.xlabel('Number of reviews',fontsize=16)
plt.ylabel('Author',fontsize=16)
plt.title('Top 10 Pitchfork writers',fontsize=16);


# ### The numbers would dictate most, if not all of these authors write Rock music reviews... 

# In[ ]:


df_final[df_final.genre=='rock'].author.value_counts()[:10].plot(kind='barh', figsize=(8,6));
plt.xlabel('Number of reviews',fontsize=16)
plt.ylabel('Author',fontsize=16)
plt.title('Top 10 Pitchfork Rock writers',fontsize=16);


# ### Yeah pretty much

# In[ ]:


df_final['author'].value_counts()[:10].sum()/df_final['author'].value_counts().sum()


# In[ ]:


df_final[df_final['genre']=='rock']['author'].value_counts()[:10].sum()/df_final['author'].value_counts().sum()


# ### So the top 10 authors write ~28% of all the reviews, while the top 10 authors in Rock write ~20% of all the reviews on the website. Quite a lot of Rock reviews I'd say. 

# ### The last 2 plots look pretty, but don't offer much by way of insight. Let's look at which authors write the longest reviews (and for which genres). 

# In[ ]:


df_final.groupby(['author','genre'])['text_words'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(8,6));
plt.xlabel('Mean review length (longest)', fontsize=16)
plt.ylabel('Author, Genre', fontsize=16);


# In[ ]:


df_final[df_final['text_len']>100].groupby(['author','genre'])['text_words'].mean().sort_values(ascending=False)[-10:].plot(kind='barh',figsize=(8,6));
plt.xlabel('Mean review length (shortest)', fontsize=16)
plt.ylabel('Author, Genre', fontsize=16);


# ### But does any of it correlate to how many unique words these authors use? Let's look at the lexical density. 

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['author','genre'])['lex_den'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(10,8),ax=ax1);
df_final[df_final['text_len']>100].groupby(['author','genre'])['lex_den'].mean().sort_values(ascending=False)[-10:].plot(kind='barh',figsize=(10,8),ax=ax2);
ax1.set_xlabel('Highest Mean Lexical Density');
ax2.set_xlabel('Lowest Mean Lexical Density');
ax1.set_ylabel('')
ax2.set_ylabel('');


# ### So the authors that write the most don't necessarily use the most unique words in their music reviews. Can we get a distinction in Lexical Density or Total words by genre?? 

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['genre'])['text_words'].mean().plot(kind='barh',figsize=(10,8),ax=ax1);
df_final.groupby(['genre'])['lex_den'].mean().plot(kind='barh',figsize=(10,8),ax=ax2);
ax1.set_xlabel('Total words');
ax2.set_xlabel('Lexical Density');
ax1.set_ylabel('')
ax2.set_ylabel('');


# ### Rap reviews tend to be longer but on average there is very little to distinguish between the different genres based on textual features. This, again, is probably a consequence of this not being an analysis of poetry or lyrics. Music reviews are typically more academic (*can we quantify this?*) so they will be more descriptive, with less flourishes that might allow for distinguishing characteristics.
# ### Is there a trend in the verbosity of reviews by year of publication?

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['pub_year'])['text_words'].mean()[::-1].plot(kind='barh',figsize=(14,12),ax=ax1);
df_final.groupby(['pub_year'])['lex_den'].mean()[::-1].plot(kind='barh',figsize=(14,12),ax=ax2);
ax1.set_xlabel('Total words');
ax2.set_xlabel('Lexical Density');
ax1.set_ylabel('')
ax2.set_ylabel('');


# ### If I were a betting man, I might say that the reviews used to be shorter, on average, in the late 90s and early 00s. However, the vocabulary used to be just as good! 
# ### Speaking of timelines, do the textual characteristics of album reviews change depending on when the album was released?

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['year'])['text_words'].mean().plot(figsize=(10,8),ax=ax1,color='b',alpha=.75);
df_final.groupby(['year'])['lex_den'].mean().plot(figsize=(10,8),ax=ax2,color='b',alpha=.75);
ax1.set_ylabel('Total words');
ax2.set_ylabel('Lexical Density');
ax1.set_xlabel('')
ax2.set_xlabel('Year of release');


# ### The reviews seem to be getting less verbose but with more unique words whenever they are about more recently released albums (Post 2000). This could do with some investigating. 
# ### Also, what is up with 1984?

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final[(df_final.year>1980) & (df_final.year<1990)].groupby(['year'])['text_words'].mean().plot(figsize=(10,8),ax=ax1,color='b',alpha=.75);
df_final[(df_final.year>1980) & (df_final.year<1990)].groupby(['year'])['lex_den'].mean().plot(figsize=(10,8),ax=ax2,color='b',alpha=.75);
ax1.set_ylabel('Total words');
ax2.set_ylabel('Lexical Density');
ax1.set_xlabel('')
ax2.set_xlabel('Year of release');


# In[ ]:


df_final[df_final.year==1984]


# In[ ]:


df_final.year.value_counts()[::-1].head(10)


# ### The 1984 case just seems like a statistical anomaly. There are just 6 reviews from that year, and they just happen to have a low ratio of unique words to total words, hence the weird spikes/dips in the curves. 
# ### And one last investigation, do review lengths/lexical density vary by artist? Let's look at the top 10 most reviewed artists

# In[ ]:


group = df_final.groupby(['artist']).agg({"text_words":"mean"}).sort_values(by='text_words',ascending=False)[:10]
group.plot(kind='barh',figsize=(10,8),color='b',alpha=.75);
plt.ylabel('Artist', fontsize=16)
plt.xlabel('Length of review (words)', fontsize=16);


# In[ ]:


group = df_final.groupby(['artist']).agg({"lex_den":"mean"}).sort_values(by='lex_den',ascending=False)[:10]
group.plot(kind='barh',figsize=(10,8),color='b',alpha=.75);
plt.ylabel('Artist', fontsize=16)
plt.xlabel('Lexical density', fontsize=16);


# ### This is a bit odd. While the longest reviews (on average) are more for reviews of more established artists, the lexical density is highest for mostly non-mainstream musicians. This suggests, perhaps, that music writers tend to be less conservative and more showy when talking about less established artists, though this is a sketchy hypothesis at best. Might be interesting to explore in the future.

# ## So that's it, there were obviously more things to explore, but I plan to have another kernel soon, where I will try and delve into, among other things, the sentiment of the review text, and whether and how that can be linked to the review score. That's it for now! Hasta la proxima!
