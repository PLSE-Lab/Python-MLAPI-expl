#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sqlite3
import pandas
with sqlite3.connect('/kaggle/input/demodataset/demoDataSet.sqlite') as connection:
    query = "SELECT * FROM `myTable`"
    data_frame = pandas.read_sql_query(query, connection)


# In[ ]:


data_frame.plot.bar(x='name',y='value')


# In[ ]:




