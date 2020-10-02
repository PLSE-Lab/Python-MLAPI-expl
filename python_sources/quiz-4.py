#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Load the dataset and display the first 5 rows

# In[ ]:


movies = pd.read_csv('/kaggle/input/movie-ratings-dataset/movie_ratings.csv').iloc[:,1:]
movies.head(5)

# .iloc[] is used to select specific cells, using the numerical index, which takes two arguments: row index, column index
# .head(n) displays the first n rows of the dataset


# Set movie names as index

# In[ ]:


movies = movies.set_index('movie')
movies.head(5)

# .set_index() is used to set a column as the index of the dataframe


# Learn how many movies are in this dataset.
# 
# There are 1800 movies in this dataset.

# In[ ]:


movies.shape[0]

# .shape gives the size of dataset, with .shape[0] displays rows and .shape[1] displays columns


# Learn the average rating score for all the movies in this dataset.
# 
# We can see that the average rating score for all the movies in this dataset is 7.24
# 

# In[ ]:


round(movies.imdb.mean(),2)

# .mean() calculates the mean of selected data


# Learn the median of the rating scores for all the movies in this dataset.
# 
# We can see that the median of all ratings are 7.3.

# In[ ]:


movies.imdb.median()

# .median() calculates the median of selected data


# Display the line plot of the distribution of ratings.
# 
# From the graph, we can see that the majority of movies are rated from around 6.8 to 8.

# In[ ]:


movies.groupby('imdb').size().plot.line()

# .groupby(x) splits data into subgroups based on feature x, i.e. split-apply-combine

# .plot.line() gives the graph, which is a line plot in this case


# Among all 1800 movies, we want to how many movies are rated above 9.
# 
# Only 28 movies are rated above 9.

# In[ ]:


movies[ movies.imdb >= 9 ].shape[0]

# movies[ movies.imdb >= 9 ] conditional selecting: select all rows with imdb >= 90


# Then we want to know how many movies are rated above 8.
# 
# There are 334 movies rated above 8.

# In[ ]:


movies[ movies.imdb >= 8 ].shape[0]


# Now, we want to analyze the data by year.
# Display the sizes of movies in each year in this dataset.
# 
# We find out that the sizes of each year are nearly equal (~100)

# In[ ]:


movies.groupby('year').size()

# .groupby.size() counts frequency - it first splits the dataset into subgroups and then count the frequencies of each subgroup


# Now, display the average rating score for each year

# In[ ]:


movies.groupby('year').imdb.mean()

# it first splits the dataset into subgroups and then calculate the mean of each subgroup


# From this list, display the top 5 years that have the highest average ratings.
# 
# We can see that 2005, 2014, 2011, 2006 and 2013 are the top 5 years, which means that these years are "big years" in the film industry. Also, it means that the competition of all kinds of film awards are intense in these years.

# In[ ]:


movies.groupby('year').imdb.mean().sort_values(ascending=False).head(5)

# .sort_values() is used to sort the data in either ascending or descending order


# Display the line plot of the ratings by year.
# 
# We can see that the trends of the change in ratings.

# In[ ]:


movies.groupby('year').imdb.mean().plot.line()


# Now, we want to learn which movie has the most ratings in this dataset.
# 
# It is 'The Dark Knight.'
# The rating of this movie is as high as 9.0, which makes sense because voting numbers and voting scores should be positively correlated.

# In[ ]:


movies.sort_values('votes', ascending=False).head(1)


# Display the movie that has the highest rating

# In[ ]:


movies.sort_values('imdb', ascending=False).head(1)


# Similarly, we can learn about which movie has the least ratings.

# In[ ]:


movies.sort_values('votes').head(1)


# Display the movie that has the lowest rating score.
# 
# It is interesting to know that 'Fifty Shades of Grey' has the lowest rating because that movie drew lots of attention in the industry at the time when it was released.

# In[ ]:


movies.sort_values('imdb').head(1)


# Now we want to discover the movies made by Adam Sandler.
# 
# First, we look for movies of Adam Sandler.

# In[ ]:


adam_sandler = movies.loc[['Mr. Deeds','Anger Management','50 First Dates','The Longest Yard','Click','Reign Over Me','I Now Pronounce You Chuck and Larry','Bedtime Stories','Funny People','Grown Ups','Just Go with It','Jack and Jill','Hotel Transylvania','Grown Ups 2','Blended','Pixels','Hotel Transylvania 2','The Ridiculous']]

# .loc[] is used to select rows using index, not numerical index 0,1,2,etc., but actual index of the dataframe


# Then we want to know the average rating of Adam Sandler's movies.
# 
# The average rating of Adam Sandler's movies is 6.39, which is lower than the average rating score.

# In[ ]:


round(adam_sandler.imdb.mean(),2)

