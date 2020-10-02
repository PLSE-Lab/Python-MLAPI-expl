#!/usr/bin/env python
# coding: utf-8

# # Split-Apply-Combine Exercises: NBA Players
# 
# The following exercises were created as a lab for the Python class I am teaching. It contains several applications of the split-apply-combine operations. (For teaching purposes, solutions are not provided.)
# 
# If you found this useful, please check out [similar exercises](https://www.kaggle.com/annieichen/split-apply-combine-exercises-goodreads-books) I created for the Goodreads Books dataset!

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


# ## Dataset
# 
# Load the dataset and perform basic cleaning. 

# In[ ]:


players = pd.read_csv('/kaggle/input/nba-players-data/all_seasons.csv', index_col=0).dropna()


# Display the first 5 rows. 

# In[ ]:





# This is more data than what we need. Update the dataset to keep only players from the most recent season (largest value in the `season` column) and only information in the first 7 columns. 

# In[ ]:





# Display the first 5 rows of the updated dataset. 

# In[ ]:





# How many players are left? 

# In[ ]:





# ## 1. Teams most diverse in age
# 
# Which are the top 10 teams with the highest standard deviation in age? 

# In[ ]:





# ## 2. Team BMI
# 
# Calculate the average body mass index (BMI) for players in each team. Display the first 5 rows. 
# 
# Recall that the BMI $= \frac{Weight (kg)}{Height (m) ^2}$. The weight and the height columns in this dataset are given in kilograms and centimeters. 

# In[ ]:





# ## 3. Countries
# 
# Create a pie plot of countries in which the players were born. 
# 
# Can you think of two solutions, one using `.groupby()` and another using `.value_counts()`? 
# - Both plots suffer from lack of readability. How might this be improved? 
# - Which plot is slightly more readable, and why? 

# In[ ]:





# ## 4. USC alumni
# 
# How many players graduated from USC? 
# 
# Can you think of two solutions, one with `.groupby()` and another without? 

# In[ ]:




