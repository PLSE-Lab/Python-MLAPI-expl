#!/usr/bin/env python
# coding: utf-8

# **Content based recommendation system:**
# 
# Content based recommendation system takes a movie name as input, then it analyzes the contents, such as genre, storyline, cast, budget, director etc., of the movie to find out similar movies.
# 
# The methodology used is cosine similarity, and it can be done using Scikit-Learn.
# 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_excel('/kaggle/input/movie-datasets/Hollywood_Movie_Dataset.xlsx')


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


# We need to convert certain features (useful for extracting the contents) to string


# In[ ]:


data[['keywords','cast','genres','director']] = data[['keywords','cast','genres','director']].astype(str)


# In[ ]:


columns = ['keywords','cast','genres','director']
def combination(n):
    return n['keywords']+" "+n['cast']+" "+n['genres']+" "+n['director']


# In[ ]:


for columns in columns:
    data[columns] = data[columns].fillna('')


# In[ ]:


data['combination'] = data.apply(combination,axis=1)


# In[ ]:


# We can use CountVectorizer() object for getting the count matrix form the combined text
count_vector = CountVectorizer()
count_matrix = count_vector.fit_transform(data['combination']) 


# In[ ]:


similarity = cosine_similarity(count_matrix)


# In[ ]:


# We need the title and the index to match for the following loop to rank movies
def index(index):
    return data[data.index == index]["title"].values[0]
def title(title):
    return data[data.title == title]["index"].values[0]


# In[ ]:


user_input = "Skyfall"
index_match = title(user_input)
similar_movies = list(enumerate(similarity[index_match]))
final_list = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Top 10 movies similar to '"+user_input+"' are:\n")
for x in final_list:
    print(index(x[0]))
    i=i+1
    if i>10:
        break


# We can see that the top 10 movies are sorted based on their similarity.

# In[ ]:




