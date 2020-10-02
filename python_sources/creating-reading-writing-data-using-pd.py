#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The first step in most data analytics projects is reading the data file. In this section, you'll create `Series` and `DataFrame` objects, both by hand and by reading data files.
# 
# # Relevant Resources
# * ** [Creating, Reading and Writing Reference](https://www.kaggle.com/residentmario/creating-reading-and-writing-reference)**
# * [General Pandas Cheat Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# 
# # Set Up
# **First, Fork this notebook using the button towards the top of the screen.  Then you can run and edit code in the cells below.**
# 
# Run the code cell below to load libraries you will need (including coad to check your answers).

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *


# # Exercises

# **Exercise 1**: Create a `DataFrame` that looks like this:
# 
# ![](https://i.imgur.com/Ax3pp2A.png)

# In[ ]:


pd.DataFrame({'Apples': [30],'Bananas': [21]})


# **Exercise 2**: Create the following `DataFrame`:
# 
# ![](https://i.imgur.com/CHPn7ZF.png)

# In[ ]:


pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]}, index = ['2017 Sales','2018 Sales'])


# **Exercise 3**: Create a `Series` that looks like this:
# 
# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```

# In[ ]:


pd.Series(['4 cups', '1 cup', '2 large', '1 can'], 
index=['Flour', 'Milk', 'Eggs', 'Spam'], 
name='Dinner')
#pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large','Spam':'1 can'}, name = "Dinner")


# **Exercise 4**: Read the following `csv` dataset on wine reviews into the a `DataFrame`:
# 
# ![](https://i.imgur.com/74RCZtU.png)
# 
# The filepath to the CSV file is `../input/wine-reviews/winemag-data_first150k.csv`.

# In[ ]:


DataF = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
print(DataF.head())


# **Exercise 5**: Read the following `xls` sheet into a `DataFrame`: 
# 
# ![](https://i.imgur.com/QZJBIBF.png)
# 
# The filepath to the XLS file is `../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls`.
# 
# Hint: the name of the method you need inclues the word `excel`. The name of the sheet is `Pregnant Women Participating`. Don't do any cleanup.

# In[ ]:


DataF = pd.ExcelFile('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
print(DataF)


# **Exercise 6**: Suppose we have the following `DataFrame`:

# In[ ]:


q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])


# Save this `DataFrame` to disc as a `csv` file with the name `cows_and_goats.csv`.

# In[ ]:


q6_df.to_csv('cows_and_goats.csv',sep = '\t')


# **Exercise 7**: This exercise is optional. Read the following `SQL` data into a `DataFrame`:
# 
# ![](https://i.imgur.com/mmvbOT3.png)
# 
# The filepath is `../input/pitchfork-data/database.sqlite`. Hint: use the `sqlite3` library. The name of the table is `artists`.

# In[ ]:


import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT * FROM artists", conn)


# ## Keep going
# 
# Move on to the **[indexing, selecting and assigning workbook](https://www.kaggle.com/residentmario/indexing-selecting-assigning-workbook)**
# 
# ___
# This is part of the [Learn Pandas](https://www.kaggle.com/learn/pandas) series.
