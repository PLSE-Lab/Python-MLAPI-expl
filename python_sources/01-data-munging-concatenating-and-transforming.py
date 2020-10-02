#!/usr/bin/env python
# coding: utf-8

# # Chapter 1- Data Munging Basics
# ## Segment 4 - Concatenating and transforming data

# In[ ]:


import numpy as np
import pandas as pd

from pandas import Series, DataFrame


# In[ ]:


DF_obj = pd.DataFrame(np.arange(36).reshape(6,6))
DF_obj


# In[ ]:


DF_obj_2 = pd.DataFrame(np.arange(15).reshape(5,3))
DF_obj_2


# ### Concatenating data

# In[ ]:


# pd.concat([left_object, right_object], axis=1)
# The concat() method joins data from seperate sources into one combined data table. If you want to 
# join objects based on their row index values, just call the pd.concat() method on the objects you 
# want joined, and then pass in the axis=1 argument. The axis=1 argument tells Python to concatenate 
# the DataFrames by adding columns (in other words, joining on the row index values).
pd.concat([DF_obj, DF_obj_2], axis =1)


# In[ ]:


pd.concat([DF_obj, DF_obj_2])


# ### Transforming data
# #### Dropping data

# In[ ]:


# object_name.drop([row indexes])
# You can easily drop rows from a DataFrame by calling the .drop() method and passing in the index 
# values for the rows you want dropped.
DF_obj.drop([0,2])


# In[ ]:


DF_obj.drop([0,2], axis=1)


# ### Adding data

# In[ ]:


series_obj = Series(np.arange(6))
series_obj.name = "added_variable"
series_obj


# In[ ]:


# DataFrame.join(left_object, right_object)
# You can use .join() method two join two data sources into one. The .join() method works by joining 
# the two sources on their row index values.
variable_added = DataFrame.join(DF_obj, series_obj)
variable_added


# In[ ]:


added_datatable = variable_added.append(variable_added, ignore_index=False)
added_datatable


# In[ ]:


added_datatable = variable_added.append(variable_added, ignore_index=True)
added_datatable


# ### Sorting data

# In[ ]:


# object_name.sort_values(by=[index value], ascending=[False])
# To sort rows in a DataFrame, either in ascending or descending order, call the .sort_values() 
# method off of the DataFrame, and pass in the by argument to specify the column index upon which 
# the DataFrame should be sorted.
DF_sorted = DF_obj.sort_values(by=[5], ascending=[False])
DF_sorted


# In[ ]:


DF_sorted = DF_obj.sort_values(by=[5], ascending=[True])
DF_sorted

