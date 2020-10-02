#!/usr/bin/env python
# coding: utf-8

# ## Analyzing the Sentiment and Subjectivity (via Textblob) of the Pitchfork Reviews dataset

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

from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

import time


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


# In[ ]:


df_final.drop(['url','author_type','pub_date','pub_weekday','pub_day','pub_month'],axis=1,inplace=True)
df_final.dropna(inplace=True)
df_final['year'] = df_final['year'].astype(int)
df_final['pub_year'] = df_final['pub_year'].astype(int)
df_final.head()


# In[ ]:


df_final.drop_duplicates('reviewid',inplace=True)
df_final['text_len'] = df_final['content'].apply(lambda x: len(x))
df_final['text_words'] = df_final['content'].apply(lambda x: len(x.split()))
df_final['uniq_words'] = df_final['content'].apply(lambda x: len(set(x.split())))
df_final['lex_den'] = 100*df_final['uniq_words']/df_final['text_words']
df_final.reset_index(inplace=True,drop=True)
df_final.head()


# ### We want to use TextBlob to find the net sentiment (positive and negative) for each review. The polarity column counts the net positive and negative polarity (TextBlob identifies polarities between -1 and +1 for each sentence, and we sum these for each review). The subjectivity column adds the subjectivity value for each sentence (as provided by TextBlob) for every sentence in a review. 

# In[ ]:


def polarity_calc(text):
    
    blob = TextBlob(text)
    pol_list = []
    for sent in blob.sentences:
        pol_list.append(sent.sentiment.polarity)
        
    return np.sum(pol_list)


# In[ ]:


def subjectivity_calc(text):
    
    blob = TextBlob(text)
    sub_list = []
    for sent in blob.sentences:
        sub_list.append(sent.sentiment.subjectivity)
        
    return np.sum(sub_list)


# In[ ]:


start = time.time()
df_final['polarity'] = df_final['content'].apply(polarity_calc)
end = time.time()
print("Time elapsed:: ",end-start,"(s)")


# In[ ]:


start = time.time()
df_final['subjectivity'] = df_final['content'].apply(subjectivity_calc)
end = time.time()
print("Time elapsed:: ",end-start,"(s)")


# In[ ]:


df_final.head()


# ### So now we have the total sentiment polarity and subjectivity (according to TextBlob) for each album review. Let's start looking at identifying some trends using this. How well do these 2 parameters correlate?

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(df_final['polarity'],df_final['subjectivity'], alpha=.25, color='b');
plt.xlabel('Polarity')
plt.ylabel("Subjectivity");


# ### There does seem to be a somewhat linear correlation between the Polarity and the Subjectivity of Pitchfork reviews. But let's check using the Pearson correlation coefficient (which is the default correlation method used by Pandas corr() method). 

# In[ ]:


df_final['polarity'].corr(df_final['subjectivity'])


# ### So the two parameters have a somewhat positive correlation. Nothing too concrete. 

# In[ ]:


df_final.groupby('genre')['polarity'].mean().plot(kind='barh', figsize=(10,8));
plt.xlabel('Polarity')
plt.ylabel('Genre');


# ### Interestingly enough, Jazz clearly seems to do the the best positive sentiment wise, while Metal does considerable poorly. This is in line with my inference from https://www.kaggle.com/delayedkarma/introductory-eda-and-genre-explorations. What about subjectivity?

# In[ ]:


df_final.groupby('genre')['subjectivity'].mean().plot(kind='barh', figsize=(10,8));
plt.xlabel('Subjectivity')
plt.ylabel('Genre');


# ### Not much to comment upon, other than the fact that Jazz and Metal both seem to have the same average level of Subjectivity! One could argue that more Subjectivity leads to greater Sentiment polarity, so the fact that these 2 genres have the most Polarized reviews makes sense! 

# ### Let's look at Sentimentality and Subjectivity for albums by date of release, overall and by genre. My intuition says the Sentimentality will be higher for older albums. 

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['year'])['polarity'].mean().plot(figsize=(10,8),ax=ax1);
df_final.groupby(['year'])['subjectivity'].mean().plot(figsize=(10,8),ax=ax2);
ax1.set_xlabel('Year');
ax2.set_xlabel('Year');
ax1.set_ylabel('Polarity')
ax2.set_ylabel('Subjectivity');


# In[ ]:


# Do overall score trends change by genre over the years?
df_final[df_final['genre']=='rock'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='b',alpha=0.75, label='Rock')
df_final[df_final['genre']=='electronic'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='r',alpha=0.75, label='Electronic')
df_final[df_final['genre']=='folk/country'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='g',alpha=0.75, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='y',alpha=0.75, label='Pop/RnB')
df_final[df_final['genre']=='jazz'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='c',alpha=0.75, label='Jazz')
df_final[df_final['genre']=='rap'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='m',alpha=0.75, label='Rap')
df_final[df_final['genre']=='experimental'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='k',alpha=0.75, label='Experimental')
df_final[df_final['genre']=='metal'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='gold',alpha=0.75, label='Metal')
df_final[df_final['genre']=='global'].groupby(['year'])['polarity'].mean().plot(figsize=(15,10),color='plum',alpha=0.75, label='Global')
plt.xlabel('Year',fontsize=16)
plt.ylabel('Mean Polarity (of Sentiment)',fontsize=16);
plt.legend(fontsize=16);


# In[ ]:


# Do overall score trends change by genre over the years?
df_final[df_final['genre']=='rock'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='b',alpha=0.75, label='Rock')
df_final[df_final['genre']=='electronic'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='r',alpha=0.75, label='Electronic')
df_final[df_final['genre']=='folk/country'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='g',alpha=0.75, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='y',alpha=0.75, label='Pop/RnB')
df_final[df_final['genre']=='jazz'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='c',alpha=0.75, label='Jazz')
df_final[df_final['genre']=='rap'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='m',alpha=0.75, label='Rap')
df_final[df_final['genre']=='experimental'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='k',alpha=0.75, label='Experimental')
df_final[df_final['genre']=='metal'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='gold',alpha=0.75, label='Metal')
df_final[df_final['genre']=='global'].groupby(['year'])['subjectivity'].mean().plot(figsize=(15,10),color='plum',alpha=0.75, label='Global')
plt.xlabel('Year',fontsize=16)
plt.ylabel('Mean Subjectivity',fontsize=16);
plt.legend(fontsize=16);


# ### My intuition was both right and wrong as it turns out! Both Polarity (Sentimentality) and Subjectivity decrease over time when all reviews are considered. Trends for Subjectivity and Sentimentality (Polarity)by genre are rather consistent over time, but there are some peaks and dips to look into. The early 90s seems to be a good period for Rap, for instance. And again, the same point I made in https://www.kaggle.com/delayedkarma/let-s-talk-about-the-text, these are music reviews, with not much scope for literary flourish, or variations in subjectivity and sentiment, really. 
# ### Let's check the distributions by genre as well

# In[ ]:


df_final[df_final['genre']=='rock']['polarity'].plot(kind='kde',figsize=(10,8),color='b',alpha=0.5, label='Rock')
df_final[df_final['genre']=='electronic']['polarity'].plot(kind='kde',figsize=(10,8),color='r',alpha=0.5, label='Electronic')
df_final[df_final['genre']=='folk/country']['polarity'].plot(kind='kde',figsize=(10,8),color='g',alpha=0.5, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b']['polarity'].plot(kind='kde',figsize=(10,8),color='y',alpha=0.5, label='Pop/R&B')
df_final[df_final['genre']=='jazz']['polarity'].plot(kind='kde',figsize=(10,8),color='c',alpha=0.5, label='Jazz')
df_final[df_final['genre']=='rap']['polarity'].plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='rap']['polarity'].plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='experimental']['polarity'].plot(kind='kde',figsize=(10,8),color='k',alpha=0.5, label='Experimental')
df_final[df_final['genre']=='metal']['polarity'].plot(kind='kde',figsize=(10,8),color='gold',alpha=0.5, label='Metal')
df_final[df_final['genre']=='global']['polarity'].plot(kind='kde',figsize=(10,8),color='plum',alpha=0.5, label='Global')
plt.xlabel('Polarity',fontsize=16)
plt.ylabel('Density',fontsize=16);
plt.legend(fontsize=16);


# In[ ]:


df_final[df_final['genre']=='rock']['subjectivity'].plot(kind='kde',figsize=(10,8),color='b',alpha=0.5, label='Rock')
df_final[df_final['genre']=='electronic']['subjectivity'].plot(kind='kde',figsize=(10,8),color='r',alpha=0.5, label='Electronic')
df_final[df_final['genre']=='folk/country']['subjectivity'].plot(kind='kde',figsize=(10,8),color='g',alpha=0.5, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b']['subjectivity'].plot(kind='kde',figsize=(10,8),color='y',alpha=0.5, label='Pop/R&B')
df_final[df_final['genre']=='jazz']['subjectivity'].plot(kind='kde',figsize=(10,8),color='c',alpha=0.5, label='Jazz')
df_final[df_final['genre']=='rap']['subjectivity'].plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='rap']['subjectivity'].plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='experimental']['subjectivity'].plot(kind='kde',figsize=(10,8),color='k',alpha=0.5, label='Experimental')
df_final[df_final['genre']=='metal']['subjectivity'].plot(kind='kde',figsize=(10,8),color='gold',alpha=0.5, label='Metal')
df_final[df_final['genre']=='global']['subjectivity'].plot(kind='kde',figsize=(10,8),color='plum',alpha=0.5, label='Global')
plt.xlabel('Subjectivity',fontsize=16)
plt.ylabel('Density',fontsize=16);
plt.legend(fontsize=16);


# ### Not much distinction according to genre. How much variation in these 2 parameters exists by author?

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['author'])['polarity'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(10,8),ax=ax1);
df_final.groupby(['author'])['subjectivity'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(10,8),ax=ax2);
ax1.set_xlabel('Polarity');
ax2.set_xlabel('Subjectivity');
ax1.set_ylabel('')
ax2.set_ylabel('');


# ### Reviews by The Picthfork Staff are quite subjective, and Simon Goddard tends to have a lot of overall positive Sentiment. 

# In[ ]:


df_final[df_final.author=='simon goddard']


# ### This turns out to be a statistical anomaly, since Simon Goddard just has one review, and for ABBA!! 

# ### One final thing:: can we relate polarity to score, by genre?

# In[ ]:


def positive_polarity_calc(text):
    
    blob = TextBlob(text)
    pos_pol_list = []
    for sent in blob.sentences:
        if sent.sentiment.polarity>0:
            pos_pol_list.append(sent.sentiment.polarity)
        
    return np.sum(pos_pol_list)

def negative_polarity_calc(text):
    
    blob = TextBlob(text)
    neg_pol_list = []
    for sent in blob.sentences:
        if sent.sentiment.polarity<0:
            neg_pol_list.append(sent.sentiment.polarity)
        
    return np.sum(neg_pol_list)


# In[ ]:


start = time.time()
df_final['positive_polarity'] = df_final['content'].apply(positive_polarity_calc)
df_final['negative_polarity'] = df_final['content'].apply(negative_polarity_calc)
end = time.time()
print("Time elapsed: ",end-start,"(s)")


# In[ ]:


df_final['pol_ratio'] = np.abs(df_final['positive_polarity']/df_final['negative_polarity'])
df_final.head()


# ### Can we correlate the polarity to the review score? What about the ratio of positive to negative polarities for a particular review?

# In[ ]:


df_final.corr()


# ### While the positive polarity relates somewhat to the score (as we would expect), the ratio of polarities is negligibly correlated.  Taking the ratio of total positive sentences to negative sentences for a particular review might be a better metric. Let's see if there are any distinctions by genre. 

# In[ ]:


# 1. Net Polarity
print("Net polarity, correlation with score")
for genre in df_final['genre'].unique():
    print(genre, df_final[df_final['genre']==genre]['score'].corr(df_final[df_final['genre']==genre]['polarity']))


# In[ ]:


# 2. Net positive polarity
print("Net positive polarity, correlation with score")
for genre in df_final['genre'].unique():
    print(genre, df_final[df_final['genre']==genre]['score'].corr(df_final[df_final['genre']==genre]['positive_polarity']))


# In[ ]:


# 2. Net negative polarity
print("Net negative polarity, correlation with score")
for genre in df_final['genre'].unique():
    print(genre, df_final[df_final['genre']==genre]['score'].corr(df_final[df_final['genre']==genre]['negative_polarity']))


# ### Well it certainly looks like negative polarity has a string correlation with the score for Metal, while the opposite is true for Jazz. Overall, the net polarity relates pretty consistently to the score across all genres, with Metal having the worst correlation, and Global music the highest (though this could just be the consequence of a very low number of reviews for the Global genre). 

# ## That's it for now. Some insight, and a lot of exploration into how TextBlob's characterization of Sentiment and Subjectivity compares to music reviews across genres, time and authors. Till next time! 

# In[ ]:




