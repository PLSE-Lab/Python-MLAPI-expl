#!/usr/bin/env python
# coding: utf-8

# **Large Dataset**

# In[ ]:


# data processing, CSV file I/O
import pandas as pd


# In[ ]:


#Load the data
data=pd.read_csv("../input/train.csv",nrows = 3000000)
print(data.shape)
data.head(3)


# In[ ]:


# Type of pickup_datetime is an object
data['pickup_datetime'].dtypes 


# In[ ]:


#convert data type of pickup_datetime object to datetime
print("Execution Time without setting the parameter infer_datetime_format as True")
get_ipython().run_line_magic('timeit', "pd.to_datetime(data['pickup_datetime'])")


# **Why to_datetime is too slow?**:-
# 
# In this instance, pandas falls back to dateutil.parser.parse for parsing the strings when no format string is supplied (more flexible, but also slower).
# 
# What does Parser mean? :-
# A parser is a compiler or interpreter component that breaks data into smaller elements for easy translation into another language. A parser takes input in the form of a sequence of tokens or program instructions and usually builds a data structure in the form of a parse tree or an abstract syntax tree. (ref.techopedia)
# 
# Enhancements :-
# pd.to_datetime learned a new infer_datetime_format keyword which greatly improves parsing perf in many cases.
# 
# * *We can improve the performance with the help of parameter  infer_datetime_format*
#  
# **infer_datetime_format** :- boolean, default False
# 
# If True and no format is given, attempt to infer the format of the datetime strings, and if it can be inferred, switch to a faster method of parsing them. In some cases this can increase the parsing speed by ~5-10x.
# 
# **timeit**:- Measure the  execution time
# 
# (ref.:pandas.pydata.org)

# In[ ]:


#Load the data again and use parameter infer_datetime_format=True
data=pd.read_csv("../input/train.csv",nrows = 3000000)
print(data.shape)
data.head(3)


# In[ ]:


# Type of pickup_datetime is an object
data['pickup_datetime'].dtypes 


# In[ ]:


#convert data type of pickup_datetime, object to datetime
# Use Parameter infer_datetime_format and set it as True
print("Execution Time with the help of parameter infer_datetime_format=True")
get_ipython().run_line_magic('timeit', "pd.to_datetime(data['pickup_datetime'],infer_datetime_format=True)")


# If we observe Execution Time without setting the parameter infer_datetime_format as True, It is around 6 min and If we set it as True, it takes few seconds  for execution.

# If anyone has the depth knowledge on infer_datetime_format. Please, share it in a comment section.

# In[ ]:




