#!/usr/bin/env python
# coding: utf-8

# **It is important to note that isnull() and isna() methods are defined in the pandas package and hence can be used for Series adn DataFrame objects only.** 

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


num_data = pd.Series([5,6,7,3,0,np.nan])


# In[ ]:


num_data.isna()


# In[ ]:


str_data = pd.Series(['Trust','Commitment','Love',None])


# In[ ]:


str_data.isna()


# In[ ]:


str_data1 = pd.Series(['Eat',np.nan,'Pray','Love','Sleep'])


# In[ ]:


str_data.isnull()


# In[ ]:


str_data.isna()


# **To check if an array contains null values, we can use isnan() method that is defined in numpy package.**

# In[ ]:


arr = np.array([3.0,4.0,5.0,np.nan,6.0])
np.isnan(arr)


# **Another alternative is to use the *in* operator that checks whether *None* or *np.nan* are present in an array or list or tuple.**

# In[ ]:


arr1 = np.array(['Fire','Ice',None,'Blood'])
None in arr1


# In[ ]:


t = ('Ed','Zed','Ted',None)
None in t


# In[ ]:


l = [1,2,3,4,4,np.nan]
np.nan in l


# In[ ]:


x = (['I','have','only',np.nan,1,'bestie',None])


# In[ ]:


(np.nan and None) in x


# In[ ]:


(np.nan or None) in x

