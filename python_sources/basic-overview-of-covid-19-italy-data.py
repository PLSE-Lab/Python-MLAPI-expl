#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This Kernel shows a few insights into the data using pandas profiler.   
# 
# Thanks to Coronavirus Emergency Response by Dipartimento della Protezione Civile (Department of Civil Protection), Italy for making the data availble to the public.
# 
# ![dpc-logo-covid19.png](attachment:dpc-logo-covid19.png)
# 

# In[ ]:


import os
import pandas as pd
import pandas_profiling 


# There are 3 csv files in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Let's first check the National Data

# In[ ]:


national = pd.read_csv('/kaggle/input/national_data.csv')
national.dataframeName = 'national_data.csv'
nRow, nCol = national.shape
print(f'There are {nRow} rows and {nCol} columns')


# ![](http://)Let's take a quick look at what the data looks like:

# In[ ]:


national.profile_report(title='National Data', progress_bar=False)


# ### Provincial Data

# In[ ]:


provincial = pd.read_csv('/kaggle/input/provincial_data.csv')
provincial.dataframeName = 'provincial_data.csv'
nRow, nCol = provincial.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


provincial.profile_report(title='Provincial Data', progress_bar=False)


# ### Regional Data

# In[ ]:


regional = pd.read_csv('/kaggle/input/regional_data.csv')
regional.dataframeName = 'regional_data.csv'
nRow, nCol = regional.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


regional.profile_report(title='Regional Data', progress_bar=False)

