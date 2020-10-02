#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

movies = pd.read_csv('../input/imdb_1000.csv', header=0)
movies.head()


# # Removing [u' , u' , and  '] with one command**

# In[ ]:


movies.actors_list.replace(['\[', 'u\'','\'\]'],'', regex=True, inplace=True)
movies.actors_list.head()


# # Removing  ' 

# In[ ]:


movies.actors_list.replace('\',', ',', regex=True, inplace=True)
movies.actors_list.unique()


# # Next, split the column into three 

# In[ ]:


actors=pd.DataFrame(movies.actors_list.str.split(',').tolist(), columns = ['actor_1','actor_2','actor_3'])
#actors.head()
movies=pd.concat([movies, actors], axis=1)
movies.head()


# #Finally, drop the combined "actors_list" column

# In[ ]:


movies.drop('actors_list', axis=1, inplace=True)
movies.sample(5)


# In[ ]:


movies.to_csv("clean_text.csv")

