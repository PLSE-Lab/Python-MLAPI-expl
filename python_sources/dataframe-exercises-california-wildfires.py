#!/usr/bin/env python
# coding: utf-8

# # DataFrame Exercises: Top 20 Largest California Wildfires
# 
# The following exercises were created as a lab for the Python class I am teaching. It covers basic operations of DataFrames including selecting, summarizing, and sorting. For teaching purposes, solutions are not provided :)
# 
# If you found this useful, please check out [similar exercises](https://www.kaggle.com/annieichen/dataframe-exercises-starbucks-drinks) I created for the Starbucks Drinks dataset! 

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
# a. Load the csv file as a DataFrame named `wildfires` and display the first 5 rows. 

# In[ ]:


# Hint: Run the previous cell to obtain the file path. 


# b. How many rows and columns does the DataFrame have? 

# In[ ]:





# c. Output the column names as a list. 

# In[ ]:





# ## 2. Selecting
# 
# a. Display only the fires since 2010. 

# In[ ]:





# b. Among the top 10 fires, how many resulted in deaths? 

# In[ ]:


# Hint: Select the relevant rows, then count the number of selected rows using `.shape`. 


# c. In which year did the Station fire occur? 

# In[ ]:


# Hint: Set `fire_name` as the index, and access the element in the row indexed `Station` and column `year`. 


# ## 3. Summarizing
# 
# a. Compute the average of acres, structures, and deaths affected by the fires. 

# In[ ]:





# b. Create a pie chart that summarizes the causes of these wildfires. 

# In[ ]:





# c. Plot the number of deaths verses acres. (Is the number of acres a good indicator of deaths?) 

# In[ ]:





# ## 4. Sorting
# 
# a. Which year does the dataset date back to (i.e., in which year did the earliest of these top fires occur)? 

# In[ ]:





# b. Display the top 5 deadliest wildfires, in descending order of the number of deaths. 

# In[ ]:





# c. Which fire affected the most structures? 

# In[ ]:




