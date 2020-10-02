#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[ ]:


data = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
data.head(5)


# In[ ]:


data = data.head(10000)


# In[ ]:


data['overview'].isnull().sum()


# In[ ]:


data['overview'] = data['overview'].fillna('')


# In[ ]:


data['overview'].isnull().sum()


# In[ ]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)


# In[ ]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


indices = pd.Series(data.index, index=data['title']).drop_duplicates()
print(indices)


# In[ ]:


idx = indices['Miracle in Milan']
print(idx)


# In[ ]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]


# In[ ]:


get_recommendations('The Frisco Kid')

