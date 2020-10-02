#!/usr/bin/env python
# coding: utf-8

# # 5 Day Data Challenge (Python) - Day 4: Bar Chart
# 
# This is Day 4 of the [5 Day Data Challenge](https://www.kaggle.com/rtatman/the-5-day-data-challenge) by Rachael Tatman. Today we will create a bar chart for categorical data.
# 
# Categorical data is data that we can assign to categories, e.g. we could have the categories 'male' and 'female', or a player in the NBA is assigned to one of the basketball teams.

# ## Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# change the plotting style
plt.style.use("fivethirtyeight")


# ## Loading the data
# 
# We read in the data with pandas.

# In[ ]:


digimon_movelist = pd.read_csv("../input/DigiDB_digimonlist.csv")
digimon_movelist.head()


# Let's examine what categories are available for the column 'Type', see also
# 
# https://chrisalbon.com/python/data_wrangling/pandas_list_unique_values_in_column/
# 
# The syntax is
# 
# `dataframe["column_name"].unique()`

# In[ ]:


print(digimon_movelist["Type"].unique())


# ## Plotting a bar chart
# 
# We can use the Pandas library to plot a bar chart, see:
# 
# https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib
# 
# The syntax is
# 
# `dataframe['column_name'].value_counts().plot(kind='bar')`
# 
# We are interested in how many digimon are in each category, i.e. how many digimon are of type `'Free', 'Vaccine', 'Virus', 'Data'` respectively.

# In[ ]:


digimon_movelist['Type'].value_counts().plot(kind='bar')

plt.title('Digimon types')
plt.ylabel('Count')
plt.show()


# We can also create the plot with the count values not sorted. For this we have to set the parameter `sort` to `False`:
# 
# `dataframe["column_name"].value_counts(sort=False).plot(kind="bar")`

# In[ ]:


digimon_movelist['Type'].value_counts(sort=False).plot(kind='bar')

plt.title('Digimon types')
plt.ylabel('Count')
plt.show()

