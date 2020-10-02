#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


rsdf=pd.read_csv('../input/resource.csv',encoding = 'utf-8') # a toy data for practice
rsdf


# There's two kinds of data indexing: `iloc` and `loc`. `iloc` is an integer-based method, while `loc` is a label-based method.
# 
# 
# Type `?pd.DataFrame.iloc` and `?pd.DataFrame.loc` for help.

# # Selections with Integers Using `iloc`

# In[ ]:


rsdf.iloc[0:6,0:4] #select multiple columns and rows


# In[ ]:


rsdf.iloc[:13,:] #select from row 0 to row 12 


# In[ ]:


rsdf.iloc[[1,3,5],[2,4]]


# In[ ]:


rsdf.loc[[1,3,5],['user_number','price']] # same as above


# In[ ]:


rsdf.iloc[2] # the information of row 2 


# # Use Conditional Selection

# In[ ]:


rsdf.loc[rsdf['source'] == 'first',:]


# In[ ]:


rsdf.loc[:,:][rsdf['source'] == 'first'] # same as above


# In[ ]:


rsdf.loc[rsdf['source'].isin(['first','third']),:] # select multple values


# # Recode

# You can flexibility recode a column by another one's row condition.

# In[ ]:


datarecode =rsdf
datarecode.loc[datarecode['source'].isin(['first','second','third']),'price']=9999.99 
datarecode

