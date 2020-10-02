#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell shows operations on dataset with ascending order (it is supposed in learntools)
import pandas as pd

columns = ['year', 'num_trips']
ascending_data = [
    [2013, 1000],
    [2014, 1001],
    [2015, 1002],
    [2016, 1003]
]
ascending_df = pd.DataFrame(data=ascending_data, columns=columns)
ascending_result = ascending_df.loc[ascending_df['year'] == 2013]['num_trips']
ascending_value = ascending_result[0]
print(ascending_result)
print(ascending_value)


# In[ ]:


# Cell shows operations on dataset with descending order (it is not supposed in learntools)
import pandas as pd

descending_data = [
    [2016, 1003],
    [2015, 1002],
    [2014, 1001],
    [2013, 1000]
]
descending_df = pd.DataFrame(data=descending_data, columns=columns)
descending_result = descending_df.loc[descending_df['year'] == 2013]['num_trips']
# Here you should use iloc[0] instead of [0]. See explanation below
descending_value = descending_result.iloc[0]
print(descending_result)
print(descending_value)


# As described in pandas docs [https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html):
# > Allowed inputs for loc are:
# A single label, e.g. 5 or 'a' (Note that 5 is interpreted as a label of the index. This use is not an integer position along the index.).
# 
# I think [] is kind of loc. So you should use iloc for position indexing.
# 
