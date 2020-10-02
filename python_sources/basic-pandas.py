#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime
import pandas.io.sql as psql
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:,.2f}'.format


# # Overview

# ## SQL
in standard SQL 
SELECT [top, distinct]
* 

FROM MAIN TABLE  
JOIN SUB TABLE  primary keys  = foreign keys 

WHERE
CONDITION

--GROUP BY  --group by must have agregation functions


SORT BY
# ######  what functions will be useful ??
a. merge
b. aggregation
c. sort
d. duplicated rows?? 
e. add use cases 
# ## Pandas and SQL 
pandas data frames are reiterable and results of which can be stored to another dataframe. This comes handy especially when using aggregated datatable = dataframe 
# # Importing DATA 

# In[ ]:


get_ipython().system('ls ../input/wine-reviews')


# In[ ]:


inpath = '../input/wine-reviews/'


# ##  from CSV

# In[ ]:


dfcsv = pd.read_csv(inpath + 'winemag-data-130k-v2.csv')


# In[ ]:


dfcsv2 = pd.read_csv(inpath + 'winemag-data_first150k.csv') 


# In[ ]:


dfcsv2.head()


# ##  from Text - Fixed Width File
pd.read_fwf()
# ##  from SQL
Needed parameters, cnxbrs = pyodbc.connect(driver='{SQL Server}',server='',database='',uid='',pwd='')
# ##  from Pickle
dfPickle = pd.read_pickle(inpath + '/wine.pickle')  #no pickle resource
# # Viewing and Cleaning Data

# ##  Head
SQL's TOP, LIMIT equivalent 
# In[ ]:


dfcsv.head() #returns top 5


# ## Tail

# In[ ]:


dfcsv.tail() #returns bottom 5


# ## Columns 

# In[ ]:


dfcsv.columns


# ## Shape
Prints dimensions of the dataframe in columns by rows (cols, rows)
# In[ ]:


dfcsv.shape


# ## Datatypes

# In[ ]:


dfcsv.dtypes


# ##  Info
Basically counts of all the columns
# In[ ]:


dfcsv.info()


# ##  Describe
# 
Statistically summary of numerical columns in the data set
# In[ ]:


dfcsv.describe()


# ##  Handling NaN

# # Filtering and Sorting

# ##  Selecting Specific Columns / Column Level

# In[ ]:


dfcsv.columns


# In[ ]:


dfcsv[['country', 'description', 'designation']]


# ##  Conditional Filtering / Row level
Filtering is index based
# ###  Single Condition filtering

# In[ ]:


dfcsv.head()


# In[ ]:


dfcsv[dfcsv.country == 'Italy']


# ###  Multiple Condition
enclose statement using "()"user may also prefer to use df.column.isin(list), rather than usign multiple conditions referring to the same column
# ####  OR
OR symbol  "|"
# In[ ]:


dfcsv[(dfcsv.country == 'Italy') | (dfcsv.region_1 == 'Etna') ]


# #### AND
AND sysbol  "&"
# In[ ]:


dfcsv[(dfcsv.country == 'Italy') & (dfcsv.region_1 == 'Etna') ]


# ###  Combination
Apply row level filtering first prior to selection of specific columns
# In[ ]:


dfcsv[(dfcsv.country == 'Italy') & (dfcsv.region_1 == 'Etna')][['country', 'region_1', 'designation']]


# ##  Duplicates

# In[ ]:


dfcsv.head()


# In[ ]:


dfcsv[dfcsv.taster_name.duplicated(keep=False)]


# ##  Sorting

# In[ ]:


dfcsv.head()


# ### Single

# In[ ]:


dfcsv.sort_values(by =['country'] , ascending = True)


# ###  Multiple

# In[ ]:


dfcsv.sort_values(by =['points', 'country' ] , ascending = [False,True])


# # Manipulating Data

# ## Adding Columns

# In[ ]:


dfcsv.head()

df['column_name'] = dfbased functions
# In[ ]:


dfcsv['points_price'] = dfcsv.points / dfcsv.price  


# In[ ]:


dfcsv[['country', 'points','price','points_price', ]].head()


# ## Lambda Functions  and Custom Functions

# ### Single Column
application of custom functions
# #### Regular Functions

# In[ ]:


def priceClass(price):
    if price > 30:
        rclass = 'Expensive'
    elif price <= 30:
        rclass = 'Cheap'
    else:
        rclass = 'Error'
    return rclass


# In[ ]:


dfcsv['priceClass'] = dfcsv.price.apply(priceClass)


# In[ ]:


dfcsv[['price','priceClass']]


# ####  Lambda Functions
lambda x : True if (x > 10 and x < 20) else False
# In[ ]:


fname = lambda point: 'Good Quality' if (point >60 ) else 'Bad Quality' 


# In[ ]:


dfcsv['Quality'] = dfcsv.points.apply(fname)


# In[ ]:


dfcsv[['points', 'Quality']]


# ###  Multiple Column

# In[ ]:


def multiCol(pClass,Qual):
    if pClass == 'Cheap' and  Qual == 'Good Quality':
        stat = 'Good Buy'
    else:
        stat = 'Dont Buy'
    
    return stat


# In[ ]:


dfcsv['Buy_NotBuy'] = dfcsv.apply(lambda df: multiCol(df.priceClass, df.Quality), axis = 1) 


# In[ ]:


dfcsv.head() #ahhh Good buy == Goodbye ahhh :')


# ## Joining Data Frames

# In[ ]:





# # Summarizing, Aggregating Data

# ##  Groupby
Data squashing according to values according to a column
# In[ ]:


dfcsv.country.value_counts()


# In[ ]:


dfcsv.groupby('country').sum()


# In[ ]:


dfcsv.groupby('country').aggregate({'points':np.sum,'price':np.mean}).sort_values(by = 'points', ascending = False)


# ##  Pivot
basically it is grouping with, column switch
# In[ ]:


dfcsv.head()


# ###  Relationship to groupby

# In[ ]:


pd.pivot_table(dfcsv, index = 'country', aggfunc={'points':np.sum,'price':np.mean}).sort_values(by = 'points', ascending = False)


# ###  Switch

# In[ ]:


pd.pivot_table(dfcsv, values =['points', 'price'] , index = 'taster_name', columns = 'country', aggfunc={'points':np.sum,'price':np.mean})


# # Exporting Data 

# ##  To CSV

# In[ ]:


dfcsv.to_csv(inpath + '/wine.csv') 


# ## To SQL

# ## To Pickle

# In[ ]:


dfcsv.to_pickle(inpath + '/wine.pickle')


# In[ ]:




