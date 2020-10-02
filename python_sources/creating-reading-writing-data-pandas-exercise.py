#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Welcome to the **[Learn Pandas](https://www.kaggle.com/learn/pandas)** track. These hands-on exercises are targeted for someone who has worked with Pandas a little before. 
# Each page it's own list of `relevant resources` you can use if you get stumped. The top item in each list has been custom-made to help you with the exercises on that page.
# 
# The first step in most data analytics projects is reading the data file. In this section, you'll create `Series` and `DataFrame` objects, both by hand and by reading data files.
# 
# # Relevant Resources
# * ** [Creating, Reading and Writing Reference](https://www.kaggle.com/residentmario/creating-reading-and-writing-reference)**
# * [General Pandas Cheat Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# 
# # Set Up
# **First, Fork this notebook using the button towards the top of the screen.  Then you can run and edit code in the cells below.**
# 
# I have forked the **[original version of this notebook](https://www.kaggle.com/residentmario/creating-reading-writing-data)** inorder to play around a bit and try completing the exercises.Thanks Kaggle and Aleksey for the awesome resources and workbook.
# 
# Run the code cell below to load libraries you will need (including coad to check your answers).

# In[1]:


import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *


# # Checking Answers
# 
# You can check your answers in each of the exercises that follow using the  `check_qN` function provided in the code cell above (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[ ]:


check_q1(pd.DataFrame())


# For the questions that follow, if you use `check_qN` on your answer, and your answer is right, a simple `True` value will be returned.
# 
# If you get stuck, you may run the `print(answer_qN())` function to print the answer outright.

# # Exercises

# **Exercise 1**: Create a `DataFrame` that looks like this:
# 
# ![](https://i.imgur.com/Ax3pp2A.png)

# In[6]:


# Your code here
pd.DataFrame({'Apples':[30],'Bananas':21},index=[0])


# Lets check if my answer is correct by invoking the custom function

# In[7]:


check_q1(pd.DataFrame({'Apples':[30],'Bananas':21},index=[0]))


# **Exercise 2**: Create the following `DataFrame`:
# 
# ![](https://i.imgur.com/CHPn7ZF.png)

# In[9]:


# Your code here
pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])


# In[10]:


#Check
check_q2(pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales']))


# **Exercise 3**: Create a `Series` that looks like this:
# 
# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```

# In[11]:


# Your code here
pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name="Dinner")


# In[12]:


#Check
check_q3(pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name="Dinner"))


# **Exercise 4**: Read the following `csv` dataset on wine reviews into the a `DataFrame`:
# 
# ![](https://i.imgur.com/74RCZtU.png)
# 
# The filepath to the CSV file is `../input/wine-reviews/winemag-data_first150k.csv`.

# In[16]:


# Your code here 
wine_review=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)


# In[17]:


#check
check_q4(pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0))


# **Exercise 5**: Read the following `xls` sheet into a `DataFrame`: 
# 
# ![](https://i.imgur.com/QZJBIBF.png)
# 
# The filepath to the XLS file is `../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls`.
# 
# Hint: the name of the method you need inclues the word `excel`. The name of the sheet is `Pregnant Women Participating`. Don't do any cleanup.

# In[19]:


# Your code here
preg=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating")


# In[20]:


check_q5(pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating"))


# **Exercise 6**: Suppose we have the following `DataFrame`:

# In[21]:


q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])


# Save this `DataFrame` to disc as a `csv` file with the name `cows_and_goats.csv`.

# In[22]:


# Your code here
q6_df.to_csv('cows_and_goats.csv')


# In[23]:


check_q6(q6_df.to_csv('cows_and_goats.csv'))


# **Exercise 7**: This exercise is optional. Read the following `SQL` data into a `DataFrame`:
# 
# ![](https://i.imgur.com/mmvbOT3.png)
# 
# The filepath is `../input/pitchfork-data/database.sqlite`. Hint: use the `sqlite3` library.

# # Additional Tip from my side
# Once we have connected to the database we need to quey the table name to get the desired output.In the tutorials part,the instructor had directly queried using the table name.Another tip could be to go to the datasources associated with this kernel and then use the table name.But here i have shown an inplace solution where using cursor() function we will be able to see all the table names associated with the database.

# In[31]:


# Your Code Here
import sqlite3
conn=sqlite3.connect('../input/pitchfork-data/database.sqlite')
#View table names in SQLite for querying:
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# In[34]:


review=pd.read_sql_query("Select * from artists",conn)


# In[35]:


review.head()


# In[36]:


#check answer
check_q7(pd.read_sql_query("Select * from artists",conn))


# ## Keep going
# 
# Move on to the **[indexing, selecting and assigning workbook](https://www.kaggle.com/residentmario/indexing-selecting-assigning-workbook)**
# 
# ___
# This is part of the [Learn Pandas](https://www.kaggle.com/learn/pandas) series.