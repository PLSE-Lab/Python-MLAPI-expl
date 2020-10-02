#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

imdb_data = pd.read_csv('../input/movie_metadata.csv') #loading the csv file
#sort to bring highest grosser movie to the top bring movie title column to front
imdb_data = imdb_data.sort_values(by = 'gross', ascending = False) 
movie_title = imdb_data['movie_title'] 
imdb_data.drop(labels=['movie_title'], axis=1,inplace = True)
imdb_data.insert(0, 'movie_title', movie_title)
imdb_data.head()


# In[ ]:


#Find how many top movies make up % of gross income

imdb_data['gross_pct'] = np.round(100 * imdb_data['gross']/imdb_data['gross'].sum(), 2)
top_10 = 100* imdb_data['gross'].iloc[:10].sum()/imdb_data['gross'].sum()
top_50 = 100* imdb_data['gross'].iloc[:50].sum()/imdb_data['gross'].sum()
top_100 = 100* imdb_data['gross'].iloc[:100].sum()/imdb_data['gross'].sum()
top_200 = 100* imdb_data['gross'].iloc[:200].sum()/imdb_data['gross'].sum()
top_500 = 100* imdb_data['gross'].iloc[:500].sum()/imdb_data['gross'].sum()
top_movies = pd.Series(data = [top_10, top_50, top_100, top_200, top_500], index = ['top_10', 'top_50', 'top_100', 'top_200', 'top_500'])
top_movies # Top 500 movies earned 50% of total gross value


# In[ ]:


director_gross = imdb_data.groupby('director_name')['gross'].sum()
director_gross = pd.DataFrame(director_gross)
director_gross = director_gross.sort_values(by = ['gross'], ascending = False)
director_gross[:10].plot(kind = 'bar')


# In[ ]:


#Find the content rating of the top 500 grosser movies
imdb_rating = imdb_data[:500]['content_rating'].value_counts()
imdb_rating.plot(kind='bar', title = "Content rating of 500 top grosser movies")


# In[ ]:




