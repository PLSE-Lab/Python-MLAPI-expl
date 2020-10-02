#!/usr/bin/env python
# coding: utf-8

# **Content Based Recommender**

# For the first part of the recommender system, we are going to build a content based recommender. We are going to extract content from overview, genres, cast, crew and keywords, vectorize the content of each film and compare similarities between films using cosine similarity. The recommender will recommend ten most similar documents given the target document.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import ast
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Getting more than one output Line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# First let's Load the datasets

# In[ ]:


dfmm=pd.read_csv('../input/movies_metadata.csv')
dfc=pd.read_csv('../input/credits.csv')
dfk=pd.read_csv('../input/keywords.csv')
dfr=pd.read_csv('../input/ratings_small.csv')
dfmm.head()
dfc.head()
dfk.head()
dfr.head()


# Drop rows with improper ids and observe the shape of each dataframe

# In[ ]:


dfmm=dfmm.drop([19730, 29503, 35587])
dfr=dfr.drop([19730, 29503, 35587])
dfmm.shape
dfc.shape
dfk.shape
dfr.shape


# Convert the datatype of feature 'id' to integer

# In[ ]:


dataframes=[dfmm, dfc, dfk]
for dataframe in dataframes:
    dataframe['id']=dataframe['id'].astype('int')


# In[ ]:


newdf=dfmm.merge(dfc, on='id')
newdf=newdf.merge(dfk, on='id')
newdf.head()


# Check null values in the features

# In[ ]:


newdf['overview'].fillna('', inplace=True)
newdf.drop(newdf[newdf['vote_average'].isnull()].index, inplace=True)


# To extract content from overviews, we tokenize each overview and apply part of speech tagging to each token. We will extract nouns, adjectives and adverbs and use only these words from overviews in our final model

# In[ ]:


def get_words(x):
    bagofwords=[]
    for i in x:
        if i[1]=='NN':
            bagofwords.append(i[0])
        elif i[1]=='NNS':
            bagofwords.append(i[0])
        elif i[1]=='NNP':
            bagofwords.append(i[0])
        elif i[1]=='NNPS':
            bagofwords.append(i[0])
        elif i[1]=='JJ':
            bagofwords.append(i[0])
        elif i[1]=='JJR':
            bagofwords.append(i[0])
        elif i[1]=='JJS':
            bagofwords.append(i[0])
        elif i[1]=='RB':
            bagofwords.append(i[0])
        elif i[1]=='RBR':
            bagofwords.append(i[0])
        elif i[1]=='RBS':
            bagofwords.append(i[0])
    return bagofwords


# In[ ]:


def clean_words(x):
    b=nltk.pos_tag(word_tokenize(x))
    result=get_words(b)
    return result


# In[ ]:


newdf['bagofwords']=newdf['overview'].apply(clean_words)


# In[ ]:


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    newdf.loc[:, feature] = newdf.loc[:, feature].apply(ast.literal_eval)


# The get_keywords function can extract the names of cast, genres and each keyword in keywords.

# In[ ]:


def get_keywords(x):
    names=[i['name'] for i in x]
    if len(names)>6:
        names=names[:6]
    return names


# The get_director_producer function can extract the names of the director and producer of the film.

# In[ ]:


def get_director_producer(x):
    names=[]
    for i in x:
        if i['job']=='Director':
            names.append(i['name'])
        elif i['job']=='Producer':
            names.append(i['name'])
    return names
            


# Transform cast, crew, keywords and genres to the forms we need. And get rid of white space between first names and last names of the all the names, combining first names with last names as single names result in stronger identifiers. For examle, for "Tom Hanks", there are many people named "Tom", but "TomHanks would be a strong indicator of whom we are referring to

# In[ ]:


features_new = ['cast', 'keywords', 'genres']
for feature in features_new:
    newdf[feature]=newdf[feature].apply(get_keywords)
newdf['crew']=newdf['crew'].apply(get_director_producer)
newdf['crew']=newdf['crew'].map(lambda x: [i.replace(" ", "") for i in x])
newdf['cast']=newdf['cast'].map(lambda x: [i.replace(" ", "") for i in x])


# Create the final "document" by combining all the content from genres, cast, crew, keywords, and bag of words

# In[ ]:


newdf['document']=newdf['genres']+newdf['cast']+newdf['crew']+newdf['keywords']+newdf['bagofwords']
newdf['document']=newdf['document'].map(lambda x: ' '.join(x))
newdf['document'].head()


# Before we feed the count vectorizer with the "document" feature. We will rule out films with too low vote_average and too low vote_count to ensure recommendation quality

# In[ ]:


pd.qcut(newdf['vote_average'], 10).values
pd.qcut(newdf['vote_count'], 4).values


# Films with vote_average less than 90 percent of all the films have vote_average less than 3.5, these films have very limited values to be recommended. Films with vote_count of less than 75 percent of all the films have vote_counts less than 49, these films have too small samples to be considered as statistically significant.

# In[ ]:


newdf=newdf[(newdf['vote_average']>3.5) & (newdf['vote_count']>34)]
newdf.reset_index(drop=True, inplace=True)


# We compute count vectors for each document. Combining this vectors will give us a matrix where each row represents a document(a movie) and each column represent a word that occurs in the overall vocabulary. The value of each cell is the count of the word occur in the document.

# In[ ]:


vectorizer = CountVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(newdf['document'])


# Compute the Cosine Similarity matrix based on the count matrix

# In[ ]:


similarity = cosine_similarity(matrix, matrix)
similarity


# In[ ]:


def recommendation(x):
    dataset=newdf.copy()
    ind=dataset[dataset['original_title']==x].index[0]
    sim=sorted(enumerate(similarity[ind]), key=lambda x: x[1], reverse=True)[1:11]
    ind2, scores=zip(*sim)
    recommendation=dataset.loc[ind2, 'original_title']
    return recommendation


# In[ ]:


recommendation('The Matrix')


# In[ ]:


recommendation('The Prestige')


# **Collaborative filtering**

# The Content based recommendation is great in recommending films based on content similarity between films. But a major flaw in content based recommender is that it does not take personal affections into account. If a user watches "The Dark Knight", it could either mean that the user likes Batman or the user likes Christopher Nolan, or it could be other reasons.

# Collaborative filtering construct a user-item matrix with user as the index and items as the columns. The value of each cell is the rating the user give the item. For a user-based filtering, we compute euclidean distance between each row to identify similar users, and recommend items similar users have watched. For a item-based filtering, we compute euclidean distance between each column to to identify similar items, and recommend items that are similar to the given item.

# In[ ]:


dfmm['id']=dfmm['id'].astype('int')
dfr['movieId']=dfr['movieId'].astype('int')


# Merge the tables with ratings and titles

# In[ ]:


titles=dfmm['original_title'].tolist()
ids=dfmm['id'].tolist()
the_map=dict(zip(ids, titles))


# In[ ]:


dfr['title']=dfr['movieId'].map(the_map)
dfr.shape


# Pivot the table so the user is the index, title is the column and value of the cell is the rating

# In[ ]:


user_item=dfr.pivot_table(index='userId', columns='title', values='rating')


# In[ ]:


user_item.fillna(0, inplace=True)


# Compute the cosine similarity of each user

# In[ ]:


user_similarity = cosine_similarity(user_item, user_item)
user_similarity


# In[ ]:


item_user=user_item.T
item_user.head()


# compute the cosine similarity of each item

# In[ ]:


item_similarity = cosine_similarity(item_user, item_user)
item_similarity


# In order to personalize the recommendation given a user U and a film I, we will get the most similar films of film I and the most similar users of the user U.  Then we average their ratings for each item of the similar films, counting only those among the similar users who have rated the item.  We will recommend only the items that have higher average ratings given by the users

# In[ ]:


def filtering_recommendations(x, a):
    dataset=item_user.copy()
    df=dataset.reset_index()
    ind=df[df['title']==x].index[0]
    sim=sorted(enumerate(item_similarity[ind]), key=lambda x: x[1], reverse=True)[1:51]
    ind2, scores=zip(*sim)
    recommendation=df.loc[ind2, 'title']
    dataset1=user_item.copy()
    df2=dataset1.reset_index()
    ind=df2[df2['userId']==a].index[0]
    sim=sorted(enumerate(user_similarity[ind]), key=lambda x: x[1], reverse=True)[1:51]
    ind2, scores=zip(*sim)
    recommendation2=df2.loc[ind2, 'userId']
    dictionary={}
    for i in recommendation.index:
        lis=[]
        for j in recommendation2.index:
            if (user_item.iloc[j, i]==0):
                continue
            else:
                lis.append(user_item.iloc[j, i])
        dictionary[i]=np.mean(lis)
    keys=[]
    for i in dictionary.keys():
        keys.append(i)
    values=[]
    for i in dictionary.values():
        values.append(i)
    ourdf=pd.Series(values, index=keys)
    sim3=ourdf.fillna(0).sort_values(ascending=False)[0:9].index
    combined_recommendation=df.loc[sim3, 'title']
    return combined_recommendation


# In[ ]:


filtering_recommendations('1984', 2)


# In[ ]:


filtering_recommendations('1984', 200)


# In[ ]:




