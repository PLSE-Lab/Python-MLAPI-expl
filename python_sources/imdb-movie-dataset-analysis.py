#!/usr/bin/env python
# coding: utf-8

# In this analysis we will take a look at the dataset provided, identify the most interesting values among it and build a plot according to it so we can visualize the information easily and come up with some conclusions.
# 
# Some interesting questions we can get from this dataset:
# 
# - How is the budget and gross profit related? This will help us understand how common is for producers to identify a good movie when they see one.
# 
# - Which countries produces the best and worst movies? Within them, which ones get closer to the high rates?
# 
# - Do people go with what they already know when it comes to movies or they try new things? Maybe analysing the director's Facebook likes and the movie's Facebook likes will give us better insights. 
# 
# - Have scores become better or worse with the years? Analysing the IMDB scores average over each decade may show us this information.
# 
# First, we need to read the file.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn#for visuals
sn.set(style="white", color_codes=True)#customizes the graphs
import matplotlib.pyplot as mp #for visuals
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings #suppress certain warnings from libraries
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
movie_data = pd.read_csv("../input/movie_metadata.csv")
movie_data


# Getting a glimpse of the size of the dataset.

# In[ ]:


movie_data.shape


# Taking a peak at the first 10 rows of data

# In[ ]:


movie_data.head(10)


#  Comparing variables to each other

# In[ ]:


movie_data.corr()


# - What does a negative value mean?
# An insignificant or non-existing relationship between the variables, their results has nothing to do with each other.
# 
# - What does a positive value mean?
# Means that there's a dependence within two variables, one may affect the other one with a lower or higher impact. 
# 
# - What does a higher number mean versus a lower number?
# It will show us if the dependency between variables is bigger or smaller, if its a high number the impact of one variable over the other one will most likely affect the result, such as the gross profit and the number of Facebook likes. Otherwise, it will mean that one variable doesn't necessarily affect the other one such as the director's Facebook likes and the movie's Facebook likes although it may seem like it.
# 
# Now, I will clean the data to prepare it for further purposes and use .describe() to understand average values so I can identify which correlation is the most interesting to analyse later.

# In[ ]:


#Cleaning the data for all the NaN values
movie_data.fillna(value=0,axis=1,inplace=True)

#Getting average values from the dataset
movie_data.describe()


# This time I will have a look at the movies' IMDB scores in comparison with the countries where they were made in order to identify which countries have the best and worst rated films. 

# In[ ]:


#Slicing the data in half for a clearer visualization
movie_sliced = movie_data[0:2501]

#Building the plot
mp.figure(figsize=(15,15))
sn.swarmplot(x='imdb_score', y='country', data = movie_sliced)
mp.title ('Which countries produce better movies?', fontsize=20, fontweight='bold')
mp.xlabel('Score')
mp.ylabel('Country')
mp.show()


# These are some conclusions we can get from the plot:
# 
# - Not surprisingly, the US and the UK are the countries with more movies produced which makes them also have a wider range for good and bad rated films.
# 
# - Apart from these two, the following countries with higher production of films are the following: Canada, Germany and France. Even though their concentration relies more on the well rated side they still have some movies that will go under 5 points of score.
# 
# - Spain, Hong Kong, China and Australia are countries with a lower production of movies but interestingly they also have all of them over the 5 points of score according to the IMDB results. 
# 
# - Apparently it seems difficult to achieve a really high rated movie because only some of the US movies will be over 9 points of score. This makes us think that audiences are not easily satisfied within the film industry.
# 
# 
