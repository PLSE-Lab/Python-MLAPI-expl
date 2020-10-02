#!/usr/bin/env python
# coding: utf-8

# # PREDICTING THE POPULARITY OF MOVIES

# ### importing required packages

# In[65]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


# ### importing data

# In[66]:


movies = pd.read_csv('../input/movies.csv')
names = pd.read_csv('../input/names.csv')
crew = pd.read_csv('../input/crew.csv')
principals = pd.read_csv('../input/principals.csv')
ratings = pd.read_csv('../input/ratings.csv')


# ### preprocessing data

# In[67]:


crew['directors'] = crew['directors'].str.split(',')
crew['writers'] = crew['writers'].str.split(',')


# In[68]:


cast = principals[principals['category'].isin(['actor', 'actress'])]     .groupby(['movie_id'])['person_id'].apply(list)     .to_frame()     .reset_index()

cast.columns = ['movie_id', 'cast']


# In[69]:


movies = movies.merge(crew, left_on='movie_id', right_on='movie_id', how='left')     .merge(cast, left_on='movie_id', right_on='movie_id', how='left')     .merge(ratings, left_on='movie_id', right_on='movie_id', how='left')


# In[70]:


movies.drop(['num_votes'], axis=1, inplace=True)


# In[71]:


movies = movies[movies['runtime'].notna()]
movies = movies[movies['avg_rating'].notna()]


# In[72]:


movies.loc[movies['directors'].isnull(), ['directors']] =     movies.loc[movies['directors'].isnull(), 'directors'].apply(lambda cell: [])
movies.loc[movies['writers'].isnull(),['writers']] =     movies.loc[movies['writers'].isnull(),'writers'].apply(lambda cell: [])
movies.loc[movies['cast'].isnull(),['cast']] =     movies.loc[movies['cast'].isnull(),'cast'].apply(lambda cell: [])


# ### data wrangling

# In[73]:


movies['popular'] = movies['avg_rating'].map(lambda cell: 1 if cell >= movies['avg_rating'].mean() else 0)


# In[74]:


def frequency_table(col_name):
    frequency_table = {}
    for row in movies[col_name]:
        for key in row:
            if key in frequency_table:
                frequency_table[key] += 1
            else:
                frequency_table[key] = 1
    return frequency_table


# In[75]:


def top(limit, data):
    top = []
    for i in range(0, limit):
        top.append(data[i][0])
    return top


# In[76]:


movies['short'] = movies['runtime'].apply(lambda cell: 1 if cell < 90 else 0)
movies['not_too_long'] = movies['runtime'].apply(lambda cell: 1 if 90 <= cell < 120 else 0)
movies['long'] = movies['runtime'].apply(lambda cell: 1 if cell >= 120  else 0)


# In[77]:


director_dic = frequency_table('directors')
director_list = list(director_dic.items())
director_list.sort(key=lambda x: x[1], reverse=True)


# In[78]:


for director in top(10, director_list):
    movies['dr' + director] = movies['directors'].apply(lambda cell: 1 if director in cell else 0)


# In[79]:


writer_dic = frequency_table('writers')
writer_list = list(writer_dic.items())
writer_list.sort(key=lambda x: x[1], reverse=True)


# In[80]:


for writer in top(10, writer_list):
    movies['wr' + writer] = movies['writers'].apply(lambda cell: 1 if writer in cell else 0)


# In[81]:


cast_dic = frequency_table('cast')
cast_list = list(cast_dic.items())
cast_list.sort(key=lambda x: x[1], reverse=True)


# In[82]:


for actor in top(20, cast_list):
    movies[actor] = movies['cast'].apply(lambda cell: 1 if actor in cell else 0)


# In[83]:


movies.drop(['movie_id', 'title', 'runtime', 'directors', 'writers', 'cast'], axis=1, inplace=True)


# ### creating model

# In[84]:


movies['is_train'] = np.random.uniform(0, 1, len(movies)) <= .75

train, test = movies[movies['is_train'] == True], movies[movies['is_train'] == False]

train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train'], axis=1, inplace=True)

train['popular'] = train['popular'].astype(int)

X_train = train.drop(labels=['popular'], axis = 1)
y_train = train['popular']

X_test = test.drop(labels=['popular'], axis = 1)
y_test = test['popular']


# In[85]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[86]:


c_dec = cross_val_score(clf, X_train, y_train, cv=10)
c_dec.mean()


# ### testing model

# In[87]:


result = clf.predict_proba(X_test)[:]
test_result = np.asarray(y_test)


# In[88]:


dec_result = pd.DataFrame(result[:,1])
dec_result['predict'] = dec_result[0].map(lambda cell: 1 if cell >= 0.6 else 0)
dec_result['answer'] = pd.DataFrame(test_result)
dec_result['correct'] = np.where((dec_result['predict'] == dec_result['answer']), 1, 0)


# In[89]:


dec_result['correct'].mean()


# ### playground

# In[90]:


'''
while True:
    print('Enter the title:')
    title = input()
    
    print('Enter the release year:')
    year = input()
    
    print('Enter the runtime in minutes:')
    runtime = input()
    
    print('Enter directors:')
    directors = input()
    
    print('Enter writers:')
    writers = input()
    
    print('Enter the cast:')
    cast = input()
'''

