#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import seaborn as sn
import matplotlib.pyplot as plt
import nltk


# In[ ]:


dataset = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
dataset.head()


# In[ ]:


import pandas_profiling
dataset.profile_report(title='Netflix Reviews - Report' , progress_bar = False)


# In[ ]:


dataset = dataset[['title','director','listed_in','description']]
dataset.head()


# finding misiing values 

# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset.director.fillna("", inplace = True)


# In[ ]:


dataset['movie_info'] = dataset['director'] + ' ' + dataset['listed_in']+ ' ' + dataset['description']


# In[ ]:


dataset.head()


# In[ ]:


dataset  = dataset[['title','movie_info']]


# In[ ]:


from nltk.corpus import stopwords
import string
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# In[ ]:


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


# In[ ]:


lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
         if i.strip().lower() not in stop:
                word = lemmatizer.lemmatize(i.strip())
                final_text.append(word.lower())
                
    return  " ".join(final_text)      
                


# In[ ]:


dataset.movie_info = dataset.movie_info.apply(lemmatize_words)


# In[ ]:


dataset.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
tf=CountVectorizer()


# In[ ]:


X=tf.fit_transform(dataset['movie_info'])  


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


cosine_sim = cosine_similarity(X)


# In[ ]:


liked_movie = 'Transformers Prime'


# In[ ]:


index_l = dataset[dataset['title'] == liked_movie].index.values[0]
similar_movies = list(enumerate(cosine_sim[index_l]))
sort_movies = sorted(similar_movies , key = lambda X:X[1] , reverse = True)
sort_movies.pop(0)
sort_movies = sort_movies[:10]


# In[ ]:


sort_movies


# In[ ]:


for movies in sort_movies:
    print(dataset.title[movies[0]])

