#!/usr/bin/env python
# coding: utf-8

# # Data types and missing data workbook
# 
# ## Introduction
# 
# This is the workbook component of the "Data types and missing data" section of the tutorial.
# 
# # Relevant Resources
# - [Data Types and Missing Data Reference](https://www.kaggle.com/residentmario/data-types-and-missing-data-reference)
# 
# # Set Up
# 
# Run the following cell to load your data and some utility functions

# In[ ]:


import pandas as pd

import seaborn as sns
from learntools.advanced_pandas.data_types_missing_data import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)


# # Checking Answers
# 
# **Check your answers in each exercise using the  `check_qN` function** (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[ ]:


check_q1(pd.DataFrame())


# If you get stuck, **use the `answer_qN` function to see the code with the correct answer.**
# 
# For the first set of questions, running the `check_qN` on the correct answer returns `True`.
# 
# For the second set of questions, using this function to check a correct answer will present an informative graph!

# # Exercises

# **Exercise 1**: What is the data type of the `points` column in the dataset?

# In[ ]:


# Your code here
import numpy as np
df = reviews.copy()
print(np.dtype())
check_q1(np.dtype(df.points))


# In[ ]:


df.head(n=10)


# **Exercise 2**: Create a `Series` from entries in the `price` column, but convert the entries to strings. Hint: strings are `str` in native Python.

# In[ ]:


# Your code here
check_q2(df.price.map(lambda x: str(x)))


# Here are a few visual exercises on missing data.
# 
# **Exercise 3**: Some wines do not list a price. How often does this occur? Generate a `Series`that, for each review in the dataset, states whether the wine reviewed has a null `price`.

# In[ ]:


# Your code here
df3 = df[df.description.notnull() == True]
#df3.price.isnull().value_counts()
check_q3(df3.price.isnull())


# **Exercise 4**: What are the most common wine-producing regions? Create a `Series` counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:
# 
# ```
# Unknown                    21247
# Napa Valley                 4480
#                            ...  
# Bardolino Superiore            1
# Primitivo del Tarantino        1
# Name: region_1, Length: 1230, dtype: int64
# ```

# In[ ]:


# Your code here
#df4 = df.copy()
#df4["region_1"] = df4["region_1"].fillna("Unknown")
#df4.groupby("region_1").region_1.count().sort_values(ascending = False)
check_q4(df.region_1.fillna("Unknown").value_counts())


# **Exercise 5**: A neat property of boolean data types, like the ones created by the `isnull()` method, is that `False` gets treated as 0 and `True` as 1 when performing math on the values. Thus, the `sum()` of a list of boolean values will return how many times `True` appears in that list.
# Create a `pandas` `Series` showing how many times each of the columns in the dataset contains null values. Your result should look something like this:
# 
# ```
# country        63
# description     0
#                ..
# variety         1
# winery          0
# Length: 13, dtype: int64
# ```
# 
# Hint: write a map that will extract the vintage of each wine in the dataset. The vintages reviewed range from 2000 to 2017, no earlier or later. Use `fillna` to impute the missing values.

# In[ ]:


# Your code here
check_q5(df.isnull().sum())
""" 
#Create string contains all the range value

year_string = ""
for i in range(2000,2018):
    #year_string+=str(i) + "+"
year_string
"""

"""
# check if "vintage" is in the description and any string in description is also in any value in year_list
year_range= range(2000,2018)
year_list = ["{}".format(year) for year in year_range]
year_list
# if any("2008" in s for s in year_list):
df[(df.description.map(lambda r: "vintage" in r) == True)&(df.description.map(lambda r: any(word in r for word in year_list) == True))]
"""
 


# # Keep going
# Move on to the [**Renaming and combining workbook**](https://www.kaggle.com/kernels/fork/598826)

# In[ ]:




