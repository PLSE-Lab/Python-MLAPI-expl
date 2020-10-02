#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np
sales_data=pd.read_csv("../input/sales_data_sample.csv",encoding='unicode_escape')


# In[3]:


sales_data.shape


# In[4]:


sales_data.head()


# In[5]:


#Find total sales yearwise

sales_data['YEAR_ID'].value_counts()


# In[6]:


sales_data.groupby(['YEAR_ID','QTR_ID']).sum()['SALES']


# In[7]:


sales_data.groupby('YEAR_ID').sum()['SALES']


# In[8]:


#Find sales Product line wise
(sales_data.groupby('PRODUCTLINE').sum()['SALES']).sort_values()


# In[9]:


#Products that they deal in

sales_data['PRODUCTLINE'].value_counts()


# In[10]:


#Country wise sales:

(sales_data.groupby('COUNTRY').sum()['SALES']).sort_values()


# In[ ]:





# This is basic data analysis using Pandas.We start off by importing the data in Sales_data dataframe.Then we check the basic information such as the shape of the dataset and info using the info method.Then we start with checking the data and columns.We go ahead and find the salesyearwise and quartewise.Then we find which product line has maximum sales.Then we check countrywise sales.We conclude that the highest sales was in 2004.Also find the country where the highest sale is recorded is USA and lowest is Ireland.We also find that Product line having maximum sales is Cars(Classic and Vintage.

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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




