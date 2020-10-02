#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

imdb = pd.read_csv('../input/movie_metadata.csv')
imdb.head()


# In[ ]:


imdb.shape


# In[ ]:


imdb.isnull().any()


# # Analysis

# 1) Finding the movie details with highest imdb score
# 
# 

# In[ ]:


imdb.groupby(['actor_1_name','actor_2_name','actor_3_name','movie_title']).imdb_score.max().sort_values(ascending = False).reset_index().head()


# 2) Analysing the budget and imdb score according to the gross of movie.

# In[ ]:


imdb.groupby(['budget','movie_title','imdb_score']).gross.max().sort_values(ascending = False).reset_index().head()


# 3) Finding the country with maximum number of movies.

# In[ ]:


imdb.country.value_counts().head()


# 4) Finding the year with maximum number of movies released.

# In[ ]:


imdb.title_year.value_counts().head()


# 5)Finding the maximum and minimum idmb score of directors of the movie.

# In[ ]:


imdb.groupby('director_name').imdb_score.agg(['max','min']).reset_index().head(10)


# 6) Inserting a coloumn to give a watching review of a movie on the basis of idmb score.

# In[ ]:


def score_review(x):
    if x<=3:
        z='Need not watch!'
    elif (x>3 and x<=7):
        z='Good to watch'
    elif x>7 and x<=10:
        z='Must watch'
    else:
        z="."
    return z
 
imdb['score_intrepretation']=imdb.imdb_score.apply(score_review)
imdb.head()


# 7) Calculating the number of times a content rating given to movies via bar graph

# In[ ]:


imdb['content_rating'].value_counts().plot(kind='bar')


# 8) Calculating the number of times a particular idmb score given to movies via bar graph

# In[ ]:


imdb['imdb_score'] = imdb['imdb_score'].apply(lambda x:int(round(x)))


# In[ ]:


imdb['imdb_score'].value_counts().plot(kind='bar')


# 9) Finding the comparison between budget and gross.

# In[ ]:


df=pd.DataFrame(imdb[['gross','budget']])
df.plot.line(subplots=True)


# 10) Analysing imdb score of different content rating

# In[ ]:


imdb.groupby('content_rating').imdb_score.max().plot(kind='bar')


# 11) Analysing the duration of movies on basis of imdb score

# In[ ]:


imdb.plot(x='duration',y='imdb_score',kind='scatter')


# In[ ]:





# In[ ]:




