#!/usr/bin/env python
# coding: utf-8

# # DataFrame Exercises: Starbucks Drinks
# 
# The following exercises were created as a lab for the Python class I am teaching. It covers basic operations of DataFrames including selecting, summarizing, and sorting. For teaching purposes, solutions are not provided :)
# 
# If you found this useful, please check out [similar exercises](https://www.kaggle.com/annieichen/dataframe-exercises-california-wildfires) I created for the Top 20 Largest California Wildfires dataset! 

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


# ## 1. DataFrame attributes
# 
# The following line loads the dataset and sets the first column as the index. 
# 
# (Note: `.dropna()` removes rows with missing values, which are represented by the string `-` in the csv file.)

# In[ ]:


drinks = pd.read_csv('/kaggle/input/starbucks-menu/starbucks-menu-nutrition-drinks.csv', index_col=0, na_values='-').dropna()


# a. Display the first 5 rows to get a sense of what the dataset looks like. 

# In[ ]:





# b. How many drinks are there?

# In[ ]:





# c. Create a new column for `calories_from_fat` (recall that one gram of fat contains 9 calories). Display the first 5 rows to see the change. 

# In[ ]:





# ## 2. Select, sort, and summarize methods
# 
# a. How many calories are in a Flat White? 

# In[ ]:





# b. How many drinks are fat-free? 

# In[ ]:





# c. Which drink contains the most fiber? 

# In[ ]:





# d. Display the average value of all columns, rounded to 2 decimal places. 

# In[ ]:





# e. Display the "Protein" and "Calories" columns for the top 10 drinks highest in protein, sorted in descending levels of protein. 

# In[ ]:





# ## 3. Plots
# 
# a. Display a histogram of calories. 

# In[ ]:





# b. Display a horizontal bar chart of the top 10 drinks highest in sodium. 

# In[ ]:





# c. Using a scatterplot, display the correlation between calories (x-axis) and calories from fat (y-axis). 

# In[ ]:





# You may notice that there is one drink that exceeds all others in both calories and calories from fat! Can you guess which one it is? Write some code to verify. 
