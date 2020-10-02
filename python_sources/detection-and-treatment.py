#!/usr/bin/env python
# coding: utf-8

# 1. ***MISSING VALUE AND ITS TREATMENT**

# In[ ]:


# import the pandas library
import pandas as pd
import numpy as np

#Using reindexing, we have created a DataFrame with missing values. 
#In the output, NaN means Not a Number.
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df


# In[ ]:


#Check for Missing Values
#Pandas provides the isnull() and notnull() functions for detecting missing value.
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

df['one'].isnull()


# In[ ]:


#Calculations with Missing Data
#When summing data, NA will be treated as Zero
#If the data are all NA, then the result will be NA
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df['one'].sum()


# In[ ]:


#Cleaning / Filling Missing Data
#Replace NaN with a Scalar Value
df = pd.DataFrame(np.random.randn(3, 3), index=['a', 'c', 'e'],columns=['one',
'two', 'three'])

df = df.reindex(['a', 'b', 'c'])

df
("NaN replaced with '0':")
df.fillna(0)


# In[ ]:


#Fill NA Forward and Backward
#pad/fill(Fill methods Forward)

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

df.fillna(method='pad')

#bfill/backfill(Fill methods Backward)
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

df.fillna(method='backfill')


# In[ ]:


#Drop Missing Values
#use the dropna function
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df.dropna()


# 2. **OUTLIER DETECTION AND ITS TREATMENT**

# In[ ]:


# import the pandas library
import pandas as pd
import numpy as np

mydata = {'productcode': ['AA', 'AA', 'AA', 'BB', 'BB', 'BB'],
'sales': [100, 1025.2, 1404.2, 1251.7, 1160, 1604.8],
'cost' : [1020, 1625.2, 1204, 1003.7, 1020, 1124]}
df = pd.DataFrame(mydata)
df          


# In[ ]:


##outlier testing
df.boxplot(column='sales')


# In[ ]:


#treatment of outliers
 
 import pandas as pd
 def remove_outlier(df):
 q1 = df['sales'].quantile(0.25)
 q3 = df['sales'].quantile(0.75)
 iqr = q3-q1 #Interquartile range
 fence_low  = q1-1.5*iqr
 fence_high = q3+1.5*iqr
 df_out = df[(df['sales'] > fence_low) & (df['sales'] < fence_high)]
    
 return df_out


# In[ ]:


df_out.boxplot(column='sales')

