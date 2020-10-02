#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as mp
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


movies_data=pd.read_csv("../input/movie_metadata.csv", sep=",",header=0) # reads in data file


# In[ ]:


movies_data.head() # displays the columun names of the dataset


# In[ ]:


movies_data.shape # shows the make up of the data set in rows and columns


# In[ ]:


movies_data.corr() # runs the correlation on the data set


# # The following is a heatmap showing the correlation within the dataset. 
# 
# The data set show a strong correlation between voted users and gross as well as voted users and critics for review and users for review.

# In[ ]:


correlation = movies_data.corr()
mp.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

mp.title('Movie Data Analysis')


# In[ ]:


chart = movies_data.groupby(["imdb_score"])['director_facebook_likes'].sum()
chart.plot(figsize=(10,5))
chart.plot()


# In[ ]:


chart = movies_data.groupby(["director_name"])['director_facebook_likes'].sum()
chart.plot(figsize=(10,5))
chart.plot()


# In[ ]:


sample_one = movies_data[['director_name','gross','imdb_score']]
sample_one.head()


# In[ ]:


sample_two = movies_data[['director_name','gross','title_year']]
sample_two.head()


# In[ ]:


sample_two_group = sample_two.groupby('title_year')
sample_two_group.size()

