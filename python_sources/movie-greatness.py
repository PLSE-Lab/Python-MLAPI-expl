#!/usr/bin/env python
# coding: utf-8

# ## Purpose ##
# The main idea here is to get some practice in working with data sets and become comfortable using python libraries for machine learning. In starting out, my goal is to create a model that can accurately predict whether a movie will be successful based upon the features provided. As I progress, this goal may expand or change altogether.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Python plotting library
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Reading and Cleanup ##
# Here we read the data from the file and cleanup the data so that it can be ready to be processed through our model down the road. I'm going to be removing the data columns that are represented by strings and only work with the data points that are represented as numbers because that's what I know how to do!

# In[ ]:


movies = pd.read_csv("../input/movie_metadata.csv") #The dataset will be represented by movie
str_columns = [] #This list will contain all columns that contain string data (to be deleted)
for colname, colvalue in movies.iteritems():
    if type(colvalue[1]) == str:
        str_columns.append(colname)
num_list = movies.columns.difference(str_columns)
movies_num = movies[num_list] #We filter out string-based data and store the remaining columns in movies_num
movies_num.head()


# In[ ]:


movies_num = movies_num.fillna(movies_num.mean()) #Replace NaN entries with their corresponding column's average value
movies_num.head()


# In[ ]:


Y = movies_num['imdb_score'] #Our output variables
X = movies_num.drop("imdb_score", 1) #Our input variables
X.head()


# ## Creating and testing different models ##
# Here we will create and test out the efficiency of different algorithms in determining output 'imdb_score"
