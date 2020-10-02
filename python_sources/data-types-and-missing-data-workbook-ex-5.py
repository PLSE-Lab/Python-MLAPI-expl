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
pd.set_option('max_rows', 10)


# In[ ]:


reviews


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


reviews.points.dtype


# **Exercise 2**: Create a `Series` from entries in the `price` column, but convert the entries to strings. Hint: strings are `str` in native Python.

# In[ ]:


reviews.price.astype(str)


# Here are a few visual exercises on missing data.
# 
# **Exercise 3**: Some wines do not list a price. How often does this occur? Generate a `Series`that, for each review in the dataset, states whether the wine reviewed has a null `price`.

# In[ ]:


reviews.price.isnull()


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


reviews.region_1.fillna("UnKnown").value_counts()


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


reviews.isnull().sum()


# # Keep going
# Move on to the [**Renaming and combining workbook**](https://www.kaggle.com/kernels/fork/598826)

# In[ ]:




