#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv("../input/Suicides in India 2001-2012.csv")
data.info()
data.tail()


# In[ ]:


data=data[ (data['Total']>0) | (data['State']!='Total (All India)') | (data['State']!='Total (States)') | 
          (data['State']!='Total (Uts)')]
data.info()


# In[ ]:


data.groupby('State').sum()['Total'].plot("barh",figsize=(13,7),title ="State wise suicides ");


# In[ ]:


data.groupby('Year').sum()['Total'].plot("bar",figsize=(13,7),title ="Year wise suicides ");


# In[ ]:


data.groupby('Gender').sum()['Total'].plot("bar",figsize=(13,7),title ="Gender wise suicides ");


# In[ ]:


data.groupby('Age_group').sum()['Total'].plot("bar",figsize=(13,7),title ="Age wise suicides ");


# In[ ]:


for type_code in data['Type_code'].unique():
    print("{0}: {1}".format(type_code, data[data['Type_code'] == type_code].size))


# In[ ]:


ds=data[data['Type_code']=='Causes']
ds.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Causes wise suicides frequency");


# In[ ]:


ds1=data[data['Type_code']=='Education_Status']
ds1.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Education_Status wise suicides frequency");


# In[ ]:


ds2=data[data['Type_code']=='Professional_Profile']
ds2.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Professional_Profile wise suicides frequency");


# In[ ]:


ds3=data[data['Type_code']=='Social_Status']
ds3.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Social_Status wise suicides frequency");


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




