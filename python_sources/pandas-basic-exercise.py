#!/usr/bin/env python
# coding: utf-8

# Pandas Basic Exercise - 

# # Pandas Data structures
# There are two types of data structure in Pandas : Series and DataFrame
# 
# Series : one dimensional data structure(ndarray), it hold a unique index.
# 
# DataFrame : two or more dimensional data structure(having table with rows and columns)

# In[ ]:


import numpy as np
import pandas as pd


# # Series

# In[ ]:


#Creating Series
lables = ['a','b','c','d','e']
my_list = [11,21,31,41,51]
l1 = ['aa','bb','cc']
# pd.Series(data=None , index = None , dtype = None , copy = False , fastpath = False) -> by default arguments for series data structure
pd.Series(data=my_list)


# In[ ]:


pd.Series(data=my_list,index=lables)


# when lable and data are not in same range then it will give error , so always data and index range should be same.

# In[ ]:


pd.Series(data=my_list,index=l1)


# In[ ]:


#chnage the dtyoe
pd.Series(data = my_list,index=lables,dtype=float)


# Operations are done based off of index:

# In[ ]:


S1 = pd.Series([1,2,3,4],index=['USA','GER','USSR','India'])
S2 = pd.Series([1,2,3,4],index=['USA','GER','Italy','India'])
S1+S2


# In[ ]:


S1-S2


# In[ ]:


S1*S2


# In[ ]:


S1/S2


# In[ ]:


S1%S2


# # DataFrame

# In[ ]:


#pd.DataFrame(data = None , index = None , columns = None , dtpye = None , copy = False)

d={'col':[1,2],'col2':[3,4]}
df=pd.DataFrame(data = d)
print(df.dtypes)
df


# In[ ]:


df = pd.DataFrame(data = d , dtype = np.int8)
print(df.dtypes)


# In[ ]:


from numpy.random import randn
np.random.seed(101)
df = pd.DataFrame(randn(5,4),index = 'A B C D E'.split(),columns = 'w x y z'.split())
df

