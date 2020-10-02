#!/usr/bin/env python
# coding: utf-8

# ## Jupyter notebook tip to display variable information - %whos 

# Using %whos magic function - we can display variable information such as values, type and data

# In[ ]:


# Variables
var1 = 100
var2 = 'This is a Test'
var3 = 2.5


# In[ ]:


# Data Structures
list_demo = [1,2,3]
dict_demo = {1:'one', 2:'two', 3:'three'}
tuple_demo = (9,8,7)


# In[ ]:


#Function
def func_demo():
    return 1


# In[ ]:


# Numpy an Pandas objects 
import numpy as np 
np_var1 = np.random.rand(1,1)
import pandas as pd 
df = pd.DataFrame


# In[ ]:


get_ipython().run_line_magic('whos', '')


# In[ ]:




