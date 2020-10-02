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
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


menu = pd.read_csv('/kaggle/input/nutrition-facts/menu.csv')


# The shape method returns a representation of the dimensionality of the DataFrame.  The output will be in a (Rows, Columns) format. 
# 
# What is the size of the menu Table?

# In[ ]:


menu.shape


# Two useful methods, tail() and sort_values() can often be used together.
# 
# - Tail() can be used to grab the last "x" number of rows in a dataframe. 
# - Sort_values() can be used sort a column from largest to smallest, or smallest to largest.  The default is smallest to largest.   To sort from largest to smallest, insert ",ascending=False" into the syntax.   
# 
# Produce the data for the ten largest items (Largest in terms of "Serving Size") using the tail and sort_values methods. 

# In[ ]:


menu.sort_values('Serving Size' ).tail(10)


# loc[] and idxmax() methods are useful for locating specific data points.  They can be used together to find data that corresponds to a maximum value.
# - The loc. method takes index labels and returns a value if the index label exists in the data frame. 
#     - Both .loc[] and .iloc[] can take two arguments: row index, column index.
# - Hint: menu.Sugars.idxmax() returns 253, which is the index of the menu item with the largest sugar quantity of all menu items.  
# 
# What menu item has the most sugar? Using idxmax and loc.

# In[ ]:


menu.loc[ menu.Sugars.idxmax() ].Item


# With the loc method, you can also locate data from an index using the set_index. After specifying the index to search from, you specifcy the value within the the index that you are looking for, as well as the column from which you want to extract data. 
# - Example: menu.set_index("Item").loc["Premium Bacon Ranch Salad with Grilled Chicken" , "Protein"]. 
# 
# This code will first search the specified "Item" column for the salad.  After the salad is found, it will return "29", which is the amount of protein in this menu item (found in the "Protein" column). 
# 
# How many calories are in the Egg McMuffin? 

# In[ ]:


menu.set_index('Item').loc['Egg McMuffin', 'Calories']


# The value_counts() method will return the sizes of each group.  
# 
# Lets pretend that we have a dataset of all students at the University of Southern California. The data set is called "students" and we have data on their class, GPA, and student ID numbers.  The code "students.Class.value_counts()" will return the number of students in each class. 
# 
# 
# Now, apply this to our McDonalds data. How many menu items are in each cateogry?

# In[ ]:


menu.Category.value_counts()


# The groupby method will group the data by the referenced column.  The mean() method will return the mean of the values.
# 
# Using these methods, what is the mean (average) calorie count of each category?  Use round() method to round each mean by two decimals.  

# In[ ]:


menu.groupby("Category").Calories.mean().round(2)


# Scatter plots can be a useful tool for analyzing the relationship between two sets of variables. 
# 
# Create a scatter plot that demonstrates the relationship between Carbohydrates and Total Fat for all menu items. 

# In[ ]:


menu.plot.scatter(x='Carbohydrates', y='Total Fat')


# Pie Charts ( .plot.pie() ) can be a useful tool for analyzing the distribution and relative size of different groupings. Create a pie chart that includes the relative proportions of each cateogry of food (Pie chart for the "Category" column using the value_counts() method.

# In[ ]:


menu.Category.value_counts().plot.pie()

