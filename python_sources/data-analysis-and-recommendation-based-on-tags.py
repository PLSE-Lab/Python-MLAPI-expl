#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df1 = pd.read_csv("../input/ted-talks/ted_main.csv")
df1.head(1)


# seems like the film_date and published_date unreadable, we need to import date time

# In[ ]:


import datetime
df1['film_date'] = df1['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df1['published_date'] = df1['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))


# In[ ]:


df1.info()


# In[ ]:


df1.nunique()


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.dropna(subset=['speaker_occupation'],inplace=True)


# In[ ]:


df1.reset_index(inplace=True)


# # Graph
# since im using just few features, i gotta make some new dataframe

# In[ ]:


df1_new = df1[['comments', 'event', 'main_speaker','title', 'speaker_occupation', 'views', 'published_date']]


# In[ ]:


df1_new.head()


# ### Speakers

# In[ ]:


fig,ax=plt.subplots(figsize=(17,5))
a=sns.barplot(y=df1_new['speaker_occupation'].value_counts(ascending=False).head(15).index, 
              x=df1_new['speaker_occupation'].value_counts(ascending=False).head(15).values, ax=ax, palette='afmhot')
a.set(title="top 15 speaker's occupation")


# #### Based on the graph we can see that the most speakers invited to give speech are writers, and it's about more than 40. assume that the writers spend more time on planning,forming, and researching the topics based what they wrote or their experienced with. the second and the third place was public figure based on art field: artists and designers. the rest of the speakers mostly is an expert on their field related with science and entertainment

# In[ ]:


fig,ax=plt.subplots(figsize=(17,6))
a=sns.barplot(y=df1_new['main_speaker'].value_counts(ascending=False).head(10).index, 
              x=df1_new['main_speaker'].value_counts(ascending=False).head(10).values, ax=ax, palette='afmhot')
a.set(title="top 10 speakers perform more than once")

df1_new[df1_new.main_speaker=='Hans Rosling'].sort_values(by='views')


# #### Hans Rosling on the first position for speakers that giving speech more than once, he had been speeched for 9 times. he was Global health expert and his first speech watched more than 12million. one of his famous creature was writen on the 'Factfulness'

# ### Events

# In[ ]:


fig,ax=plt.subplots(figsize=(17,5))
b=sns.barplot(y=df1_new['event'].value_counts(ascending=False).head(15).index, 
            x=df1_new['event'].value_counts(ascending=False).head(15).values, ax=ax, palette='afmhot')
b.set(title='top 15 most held yearly event')


# #### Based on event, TED2014 and TED2009 held more than 80 times, but accumulative in general year TED had been held more than 300 times in 2012 (graph time series below)

# In[ ]:


df1_new['year']=df1_new['published_date'].apply(lambda x: x[-4:])
ig,ax=plt.subplots(figsize=(17,7))
sns.lineplot(x=df1_new['year'].value_counts().index,y=df1_new['year'].value_counts().values, marker='o')
sns.set_style('darkgrid')


# # Views and Comments

# In[ ]:


fig,ax=plt.subplots(figsize=(17,7))
d=sns.barplot(x=df1_new[df1_new['views']>1000000].sort_values(by='views', ascending=False).head(10)['views'],
              y=df1_new[df1_new['views']>1000000].sort_values(by='views', ascending=False).head(10)['title'],palette='afmhot', ax=ax)
d.set(xlim=(15000000,50000000))
d.set(title='top 10 most watched based on title')


# #### Based on the views, the title look interesting and extraordinary. most of the title tags appeared is about culture

# In[ ]:


fig,ax=plt.subplots(figsize=(17,7))
e=sns.barplot(y=df1_new.sort_values(by='comments', ascending=False)['title'].head(10).values,
            x=df1_new.sort_values(by='comments', ascending=False)['comments'].head(10).values, palette='afmhot', ax=ax)
e.set(title='top 10 most commented based on title')


# #### most of the tags appeared are about culture and psycology. Based on the most commented is about faith or what people believe ('Militant atheism') total commented more than 6000 (it doesnt disclose the type of the comment: is it a respond to the TED video or a respond to another comment from another user)

# # Recommendation System based on Tags

# ### clean the tags

# In[ ]:


import re


# In[ ]:


def clean_text(x):
    letter_only=re.sub("[^a-zA-Z]", " ", x)
    return ' '.join(letter_only.split()).lower()


# In[ ]:


df1_new['tags']=df1['tags']
df1_new.tags=df1_new.tags.astype('str')


# In[ ]:


df1_new['tags']=df1_new['tags'].apply(clean_text)


# In[ ]:


df1_new.head(1)


# ### Convert to sparse matrix using count vectorize

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv=CountVectorizer()
cv_tags=cv.fit_transform(df1_new['tags'])
df_genres=pd.DataFrame(cv_tags.todense(), columns=cv.get_feature_names(), index=df1_new['title'])


# ### using Cosine Similarity

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


cos_sim=cosine_similarity(cv_tags)


# In[ ]:


def get_recommendation_based_title(x):
    index_to_search = df1_new[df1_new['title']==x].index[0]
    series_similar=pd.Series(cos_sim[index_to_search])
    index_similar=series_similar.sort_values(ascending=False).head(10).index
    return df1_new.loc[index_similar]


# In[ ]:


get_recommendation_based_title('Do schools kill creativity?')


# In[ ]:


def get_recommendation_based_speakers(x):
    index_to_search = df1_new[df1_new['main_speaker']==x].index[0]
    series_similar=pd.Series(cos_sim[index_to_search])
    index_similar=series_similar.sort_values(ascending=False).head(10).index
    return df1_new.loc[index_similar]


# In[ ]:


get_recommendation_based_speakers('Hans Rosling')


# In[ ]:


def get_recommendation_based_speaker_occupation(x):
    index_to_search = df1_new[df1_new['speaker_occupation']==x].index[0]
    series_similar=pd.Series(cos_sim[index_to_search])
    index_similar=series_similar.sort_values(ascending=False).head(10).index
    return df1_new.loc[index_similar]


# In[ ]:


get_recommendation_based_speaker_occupation('Artist')


# #### Based on 3 function above, we can get reccomendation using title, main speakers and speakers occupation based on tags on the video. to get the similarity tags we are using count vectorizer to make the binary, and then transform it into numbers that can be count based on their cosine similarity to get their nearest tags
