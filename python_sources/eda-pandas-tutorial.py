#!/usr/bin/env python
# coding: utf-8

# ***The following will be an exploratory analysis of the dataset along with a tutorial of Pandas***
# 
# **Hey there!** You have been just been hired by an AI company that is looking to optimize restraurants based on their reviews so they can best recommend customers which restaurant to go to. In the following exercises we will look at this dataset to see how we can draw insights into how we should recommend restaurants.

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


# *first things first let's import pandas, matplotlib, and the dataset*

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/restaurant-week-2018/restaurant_week_2018_final.csv')


# ## 1. Lets find out what we have in this datset in the first place.

# *a) Find the first 5 rows of the data*

# In[ ]:


data.head()


# *b) Find how many rows are in our data set*

# In[ ]:


data.shape[0]


# c) *let's get our feet wet with selecting rows of the dataset, select and display rows 200-210*
# 
# PS: we'll be using selection later on!

# In[ ]:


data.iloc[200:211]


# *d) the last thing we need to do before diving deeper into this dataset is find the highest and lowest reviewed restaurant*
# 
# PS: you need to use .loc or .iloc for this!

# grab the name of the highest reviewed restaurant:

# In[ ]:


data.loc[data.average_review.idxmax()][0]


# lowest reviewed:

# In[ ]:


data.loc[data.average_review.idxmin()][0]


# ## 2. Awesome job, now let's move on to some more complex analysis

# *a) let's find out which restaurants have more than 1000 reviews*

# In[ ]:


data.query('review_count > 1000')


# *b) now lets find some summary statistics for restaurants with more than 1000 reviews and less thean 1000 reviews*

# More than 1000

# In[ ]:


data.query('review_count > 1000').mean()


# Less than 1000

# In[ ]:


data.query('review_count < 1000').mean()


# *c) now let's look at each restuarant type and find averages for type to see if we can find any trends there*

# In[ ]:


data.groupby("restaurant_main_type").mean()


# *c) part 2) how many of each restaurant type is represented in the dataset?*

# In[ ]:


data.groupby("restaurant_main_type").size().sort_values(ascending=False)


# *d) to help us find some trends let's sort out the grouped dataset by value review in descending order*

# In[ ]:


data.groupby("restaurant_main_type").mean().sort_values('value_review', ascending = False)


# *e) in order to visually see what the difference between value of different cuisines is let's plot it*
# 
# hint: you may need to create a new dataframe to plot the chart

# In[ ]:


plotData = data.groupby("restaurant_main_type").mean().sort_values('value_review', ascending = False).value_review
plotData.plot.barh()


# *f) next let's make a chart with average reviews for each cuisine*

# In[ ]:


plotData2 = data.groupby("restaurant_main_type").mean().sort_values('value_review', ascending = False).average_review
plotData2.plot.barh()


# **With all these learnings in mind, what do you think we should prioritize in rankings for our AI?**

# Potential Answer: It is important to have many variables for the AI, however you can weigh them based on deviation and amount of resposnses to have the most thorough and highest success results!
