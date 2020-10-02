#!/usr/bin/env python
# coding: utf-8

# # Using Metadata with Word2Vec to get recommendations in MovieLens

# The idea of this notebook is to introduce an idea to use Word2Vec in a well-known dataset such as MovieLens. The approach is to define documents guided by what each user saw. After train we can relate films and its features. Also users can be related to them.

# In[ ]:


import numpy as np
import pandas as pd


# First we'll get the data and sort by time. The reason of sorting by time is due to the fact that we need the movies that each user saw in chronological order.

# In[ ]:


df_train = pd.read_csv("../input/u.data.csv", names=['user_id', 'item_id', 'ranking', 'time'], sep='\t')
df_train['time'] = pd.to_datetime(df_train['time'],unit='s')
df_train = df_train.sort_values(by='time')


# And filter with ranking above of 3 points

# In[ ]:


df_train = df_train[df_train['ranking'] > 3]


# In[ ]:


df_train.head()


# Getting item features

# In[ ]:


names=['item_id', 'movie title', 'release date', 'video release date',
              'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
              'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western' ]
df_items = pd.read_csv("../input/u.item", names= names, sep='|', encoding = 'ISO-8859-1')


# In[ ]:


df_items = df_items.filter(['item_id', 'movie title', 'Action', 'Adventure', 'Animation',
              'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western'])


# In[ ]:


df_items.head()


# Merge train with items to get their genre

# In[ ]:


df_train_2 = pd.merge(df_train, df_items, on='item_id')


# Let's see if how the train dataframe was built. We want to see all the items watched by user_id 914

# In[ ]:


df_train_2[df_train_2['user_id'] == 914]


# One film can be in several genres at once. We can see this adding a column 'Total' that is the sum of all the genres. 

# In[ ]:


l = ['Action', 'Adventure', 'Animation', 
    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western']
    
def f(row):
    sum = 0
    for i in l:
        sum = sum+row[i] 
    return sum
    
df_train_2['Total'] = df_train_2.apply(f, axis=1)
df_train_2.head()


# In the following cell, we can see that the majority of the films have 2 features. In the next steps we will see why this is important

# In[ ]:


df_items['Total'] = df_items.apply(f, axis=1)
df_items['Total'].describe()


# Now we will create a dataset with all the users and all their watched films sorted by timestamp. Also we'll add the first genre that is related to each movie. I know that a movie has several genres but this is to keep the example simple. Undoubtedly this could be coded better.

# In[ ]:


def get_features_movie(item_id):
    l = ['Action', 'Adventure', 'Animation', 
    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western']
    
    features = []
    temp = df_items[df_items['item_id'] == item_id]
    for i in l:
        if temp.iloc[0][i]:
            features = features + [i]
    
    return features

def convert_to_list(item_id):
    return " ".join([str(x) for x in get_features_movie(item_id)[:1]])

train_watched = pd.DataFrame(columns=['user_id', 'watched'])

for index, user_id in enumerate(range(min(df_train_2['user_id']), max(df_train_2['user_id']))):
    d = df_train_2[df_train_2['user_id'] == user_id].filter(['item_id'])
    l = d['item_id'].tolist()
    l_to_string = " ".join([convert_to_list(x)+" "+str(x) for x in l])
    train_watched.loc[index] = [user_id, l_to_string]


# In[ ]:


train_watched.head()


# Therefore in the new dataset we can see the trace of watched films and its genre for each user. It's time to use word2vec.

# In[ ]:


from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec


# In[ ]:


list_doc = []

for row in train_watched.to_dict(orient='record'):
    list_doc.append(str(row['watched']).strip().split(' '))


# In[ ]:


model = Word2Vec(list_doc, window=5, min_count=1, workers=4)


# After train we have to see the results.

# In[ ]:


def most_similar(item_id_or_genre):
    try:
        print("Similar of "+df_items[df_items['item_id'] == int(item_id_or_genre)].iloc[0]['movie title'])
    except:
        print("Similar of "+item_id_or_genre)
    return [(x, df_items[df_items['item_id'] == int(x[0])].iloc[0]['movie title']) for x in model.wv.most_similar(item_id_or_genre)]


# In[ ]:


most_similar('Action')


# It seems that it works! The query of the word 'Action' is near to action movies such as 'GoldenEye (1995)'. What about 'Horror'?

# In[ ]:


most_similar('Horror')


# All of the movies listed are about Horror. Thumbs up! :)

# Let's see movies related with Die Hard 2...

# In[ ]:


most_similar('226')


# All the results are almost action, and also Die Hard 3 is recommended. Quite good! This could be used when you want a recommendation from a genre or an item. But what about a recommendation for a user?. For example what films the user_id 914 saw?

# In[ ]:


df_train_2[df_train_2['user_id'] == 914].filter(['item_id', 'movie title']+l)


# Let's define a user as a vector of the average of movies that he/she saw.

# In[ ]:


def create_avg_user_vector(user_id):
    item_id_list = df_train_2[df_train_2['user_id'] == user_id]['item_id'].tolist()
    vector_item_id_list = [model.wv[str(x)] for x in item_id_list]
    return np.average(vector_item_id_list, axis=0)

def most_similar_by_vector(vector):
    return [(x, df_items[df_items['item_id'] == int(x[0])].iloc[0]['movie title']) for x in model.wv.similar_by_vector(vector)]

most_similar_by_vector(create_avg_user_vector(914))


# Analysing the results we can see that almost the movies are comedies or family films. For instance 'First Kid (1996)' and 'Father of the Bride Part II (1995)' are definetly comedie films. Also the last one it's romantic too. So, we can describe an user using their movies preferencies. 

# ## Some considerations
# * The idea of use the first genre of the table is completely arbitrary. I tried doing the same adding two genres to see if it could be better. It seems that it's not working notoriusly better. Also using all the genres in the documents imples that the window size in the model must be tunned. It could be nice for a future work of this notebook.
# * It is possible that genre that are in the last columns could not be in the vocabulary. That is because we get only the first genre.
# * This ideas became after watching the following video https://www.youtube.com/watch?v=TINZK94reEE. In that video, Flavian said that he could perform recommendation without adding a new implementation of word2vec. I suspect that he did something similar of this. Also there is a paper about it: https://arxiv.org/pdf/1607.07326.pdf
# * The idea of represent an user with an average is from https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484 
# * If someone read this article and have advice of improve the code, it is completely welcomed!
# 
# Thanks for reading! :)
