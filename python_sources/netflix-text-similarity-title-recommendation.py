#!/usr/bin/env python
# coding: utf-8

# # # NETFLIX - SIMILAR [MOVIE/SHOW] RECOMMENDATION 

# ******The aim of this note book is Text similarity using NLP . The practical application includes usage of Tockenizing, Stemming, Vectorizing and Similarity finding . 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # # # #  **Encapsulating the Imports**

# *Sklearn and **rake_nltk** are the special packages that will help us find the similar title . Recently researched about rake_nltk and Rake function to extract key words from the description column, which inturn helps us not using the entire description, rather finding root words or key words that help.*

# In[ ]:


import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# # # # Read the csv

# In[ ]:


df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

# Iam dropping all nan since that is out of scope and also i have ample dataset after the drop - 
# you can use any means to fill na - 
df = df.dropna()


# In[ ]:


df = df[['title','type','director','cast','description']]
df.head()


# # # # Data cleaning 

# It's important to clean the data specially when it comes to text data. Here, as we try to find the similar movies or tv shows, we also need to be aware of the columns contributing to the goal and how well the contributing columns are cleaned to give us good results . Here we take into account, the cast, director and type of movie columns to be split to seperate words 
# 
# * Merging together first and last name for each actor and director, so it's considered as one word 
# * To Ensure there isnt any collaboration in people sharing their first names
# * Also listing out the cast,director,type to be list of root words

# In[ ]:


df['cast'] = df['cast'].map(lambda x: str(x).split(',')[:3])


# In[ ]:


df['type'] = df['type'].map(lambda x: x.lower().split(','))


# ******Clean up for director column  **

# In[ ]:


df['director'] = df['director'].map(lambda x: str(x).split(' '))


# In[ ]:


for index, row in df.iterrows():
    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
    row['director'] = ''.join(row['director']).lower()


# # # Making use of **Rake** function 
# * To get the keywords and scores as dictionary
# * Initializing to a new column for each corresponding movie 
# 

# In[ ]:


df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['description']
    
    # It uses english stopwords from NLTK ,discards punctuation
    r = Rake()

    # extracting root words
    r.extract_keywords_from_text(plot)

    # keywords as keys and scores as values in dict
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())


# In[ ]:


# dropping the description column
df.drop(columns = ['description'], inplace = True)


# In[ ]:


# set movie title as index
df.set_index('title', inplace = True)
df.head()


# *As we collected all the necessary root words, lets combine to bag of words that has all keywords from all the significant columns for us to predict*

# # # Bag of words 

# In[ ]:


# Bag of words
df['final_keywords'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['final_keywords'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'final_keywords'], inplace = True)


# # # Count Vectorizer 

# ***CountVectorizer** to create a Series for the movie titles so they are associated to an ordered numerical*

# In[ ]:


# Initialize the frequency counts
count = CountVectorizer()
count_matrix = count.fit_transform(df['final_keywords'])


# In[ ]:


# Creating series
indices = pd.Series(df.index)
indices[:5]


# # # Cosine Similarity

# In[ ]:


# generating the cosine similarity to find the vector coordinance
cosine_ = cosine_similarity(count_matrix, count_matrix)
cosine_


# # # Recommendations

# In[ ]:


# function that takes in movie title as input and returns the top 10 recommended movies 
def recommended_netflix(title, cosine_sim = cosine_):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    scores = pd.Series(cosine_[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_movies = list(scores.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_movies:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies


# # # Sample testing

# In[ ]:


# find the similar shows or movies with respect to similarity 
recommended_netflix('Norm of the North: King Sized Adventure')


#  **This notebook is inspired by -Emma Grimaldi - by the article came to know about rake function
