#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


movie = pd.read_csv("../input/movie.csv")
movie.columns


# In[ ]:


movie = movie.loc[:,["movieId","title"]]
movie.head(10)


# In[ ]:


# import rating data and look at columsn
rating = pd.read_csv("../input/rating.csv")
rating.columns


# In[ ]:


# what we need is that user id, movie id and rating
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)


# In[ ]:


# then merge movie and rating data
data = pd.merge(movie,rating)


# In[ ]:


# now lets look at our data 
data.head(10)


# * Lets look at shape of the data. The number of sample in data frame is 20 million that is too much. There can be problem in kaggle even if their own desktop ide's like spyder or pycharm.
# * Therefore, in order to learn item based recommendation system lets use 1 million of sample in data.

# In[ ]:


data.shape


# In[ ]:


data = data.iloc[:1000000,:]


# In[ ]:


# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)


# * Now lets make a scenario, we have movie web site and "Bad Boys (1995)" movie are watched and rated by people. The question is that which movie do we recommend these people who watched "Bad Boys (1995)" movie.
# * In order to answer this question we will find similarities between "Bad Boys (1995)" movie and other movies.

# In[ ]:


movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
similarity_with_other_movies.head()


# * It can be concluded that we need to recommend "Headless Body in Topless Bar (1995)" movie to people who watched "Bad Boys (1995)".
# * On the other hand even if we do not consider, number of rating for each movie is also important.

# 

# In[ ]:


get_ipython().system('pip install turicreate')


# In[ ]:




