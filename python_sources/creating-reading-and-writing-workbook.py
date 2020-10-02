#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Welcome to the **[Learn Pandas](https://www.kaggle.com/learn/pandas)** track. These hands-on exercises are targeted for someone who has worked with Pandas a little before. 
# Each page has a list of `relevant resources` you can use if you get stumped. The top item in each list has been custom-made to help you with the exercises on that page.
# 
# The first step in most data analytics projects is reading the data file. In this section, you'll create `Series` and `DataFrame` objects, both by hand and by reading data files.
# 
# # Relevant Resources
# * ** [Creating, Reading and Writing Reference](https://www.kaggle.com/residentmario/creating-reading-and-writing-reference)**
# * [General Pandas Cheat Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# 
# # Set Up
# 
# Run the code cell below to load libraries you will need (including code to check your answers).

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")


# # Exercises

# ## 1.
# 
# In the cell below, create a DataFrame `fruits` that looks like this:
# 
# ![](https://i.imgur.com/Ax3pp2A.png)

# In[ ]:


# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame({"Apples":[30],"Bananas":[21]})

q1.check()
fruits


# In[ ]:


# Uncomment the line below to see a solution
#q1.solution()


# ## 2.
# 
# Create a dataframe `fruit_sales` that matches the diagram below:
# 
# ![](https://i.imgur.com/CHPn7ZF.png)

# In[ ]:


# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame({"Apples":[35,41],"Bananas":[21,34]},index = ["2017 Sales","2018 Sales"])

q2.check()
fruit_sales


# In[ ]:


#q2.solution()


# ## 3.
# 
# Create a variable `ingredients` with a `pd.Series` that looks like:
# 
# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```

# In[ ]:


ingredients = pd.Series(["4 cups","1 cup","2 large","1 can"],index = ["Flour","Milk","Eggs","Spam"], name = "Dinner")

q3.check()
ingredients


# In[ ]:


#q3.solution()


# ## 4.
# 
# Read the following csv dataset of wine reviews into a DataFrame called `reviews`:
# 
# ![](https://i.imgur.com/74RCZtU.png)
# 
# The filepath to the csv file is `../input/wine-reviews/winemag-data_first150k.csv`.

# In[ ]:


reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col = 0)

q4.check()
reviews


# In[ ]:


#q4.solution()


# ## 5.
# 
# Run the cell below to create and display a DataFrame called `animals`:

# In[ ]:


animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals


# In the cell below, write code to save this DataFrame to disk as a csv file with the name `cows_and_goats.csv`.

# In[ ]:


# Your code goes here
animals.to_csv("cows_and_goats.csv")
q5.check()


# In[ ]:


#q5.solution()


# ## 6.
# 
# This exercise is optional. Read the following SQL data into a DataFrame called `music_reviews`:
# 
# ![](https://i.imgur.com/mmvbOT3.png)
# 
# The filepath is `../input/pitchfork-data/database.sqlite`. Hint: use the `sqlite3` library. The name of the table is `artists`.

# In[ ]:


import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
music_reviews = pd.read_sql_query("select * from artists",conn)
q6.check()
music_reviews


# In[ ]:


#q6.solution()


# ## Keep going
# 
# Move on to the **[indexing, selecting and assigning workbook](https://www.kaggle.com/kernels/fork/587910)**
# 
# ___
# This is part of the [Learn Pandas](https://www.kaggle.com/learn/pandas) series.
