#!/usr/bin/env python
# coding: utf-8

# Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import nltk


# Importing the DataSets 

# In[ ]:


movies_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')


# Recommendation Using Content Based Filtering

# Merging the two DataSets
# 

# In[ ]:


data = movies_data.merge(credits_data, how = 'inner', on = movies_data['id'])


# In[ ]:


data.info()


# Dropping the Redundant Columns

# In[ ]:


data.drop(['homepage', 'key_0', 'title_x', 'title_y', 'movie_id'], axis = 1, inplace = True)


# In[ ]:


data.info()


# In[ ]:


from ast import literal_eval


# Applying the literal_eval on the Columns having Unstructured Data

# In[ ]:


data['genres'] = data['genres'].apply(literal_eval)


# In[ ]:


data['keywords'] = data['keywords'].apply(literal_eval)


# In[ ]:


data['cast'] = data['cast'].apply(literal_eval)


# In[ ]:


data['crew'] = data['crew'].apply(literal_eval)


# Function to get the Director's Name

# In[ ]:


def getDirectorName(crew) :
    for i in crew :
        if i['job'] == 'Director' :
            return i['name']


# Function to get the top 5 Keyword, Genres and Keyword

# In[ ]:


def getNameList(words) :
    names = []
    for i in words :
        names.append(i['name'])
    
    if len(names) > 5 :
        return names[0:5]
    
    else :
        return names


# In[ ]:


data['Director'] = data['crew'].apply(lambda x : getDirectorName(x))


# In[ ]:


data['keywords'] = data['keywords'].apply( lambda x : getNameList(x) )


# In[ ]:


data['genres'] = data['genres'].apply( lambda x : getNameList(x) )


# In[ ]:


data['cast'] = data['cast'].apply( lambda x : getNameList(x) )


# In[ ]:


data.info()


# Filling the Null Values of the Director Column with ' '

# In[ ]:


data['Director'] = data['Director'].fillna(' ')


# Removing the whitespaces in the Director name and converting all to lowercase

# In[ ]:


def cleanDiretor(name) :
    name = name.lower()
    name = name.replace(' ', '')
    return name


# Removing the spaces in the names and converting all to lowercase

# In[ ]:


def cleanWordList(words) :
    names = []
    
    for word in words :
        word = word.lower()
        word = word.replace(' ', '')
        names.append(word)
    return names


# Combining the Director Name, Cast, Keywords and Genres in a single column

# In[ ]:


def finalData(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['Director'] + ' ' + ' '.join(x['genres'])

data['Final_Data'] = data.apply(finalData, axis=1)


# In[ ]:


data.info()


# Importing the CountVectorizer and creating BOW matrix

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


countvectorizer = CountVectorizer()


# In[ ]:


vectormatrix = countvectorizer.fit_transform(data['Final_Data'])


# In[ ]:


vectormatrix.shape


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similaritymatrix = cosine_similarity(vectormatrix, vectormatrix)


# In[ ]:


similaritymatrix.shape


# In[ ]:


similaritymatrix[0][0:10]


# Creating a Series of Indices and Title

# In[ ]:


indices = pd.Series(data = data['id'].index, index = data['original_title']).drop_duplicates()


# In[ ]:


indices


# Creating a Function to recommend movies based on a movie

# In[ ]:


def getSimilarMovies(moviename) :
    index = indices[moviename]
    
    similarmovies = list(enumerate(similaritymatrix[index]))
    similarmovies = sorted(similarmovies, key = lambda x : x[1], reverse = True)
    similarmovies = similarmovies[1:11]
    
    moviesindex = []
    
    for movie in similarmovies :
        moviesindex.append(movie[0])
        
    similarmovies = data['original_title'].iloc[moviesindex]
    
    return similarmovies


# Getting Movies Similar to the movie 'The Wolverine'

# In[ ]:


getSimilarMovies('The Wolverine')


# Getting Movies Similar to the movie 'The Dark Knight Rises'

# In[ ]:


getSimilarMovies('The Dark Knight Rises')


# In[ ]:




