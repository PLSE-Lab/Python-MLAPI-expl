#!/usr/bin/env python
# coding: utf-8

# In[11]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **DC**

# In[12]:


# Read data DC and Marvel
dc = pd.read_csv('../input/dc-wikia-data.csv')
dc.head(3)


# In[13]:


mr = pd.read_csv('../input/marvel-wikia-data.csv')
mr.head(3)


# In[14]:


# We can control whether data has NaN value or not, the code below 
# assert pd.notnull(df).all().all()


# In[15]:


# Look at the info; data types, non-null entry numbers, range index and memory usage.
dc.info()


# In[16]:


# We can see number of NaN values of the features:
dc.isna().sum()


# In[17]:


# GSM feature is not usable for me, because there are only 64 values. So we can drop:
dc = dc.drop('GSM', axis = 1)
dc.head(3)


# In[18]:


# String characters should be regulated. If there are any space character there will be problem.
for col in dc.select_dtypes([np.object]):
    dc[col] = dc[col].str.strip()


# In[20]:


# We search for categorical data. 'ID', 'SEX', 'ALIVE' are categorical. We should change types of this features. 
# Memory usage will decrease and usability will increase.
dc['ID'].value_counts()


# In[21]:


dc['ID'] = dc['ID'].astype('category')


# In[22]:


dc['SEX'].value_counts()


# In[23]:


dc['SEX'] = dc['SEX'].astype('category')


# In[24]:


dc['ALIVE'].value_counts()


# In[25]:


dc['ALIVE'] = dc['ALIVE'].astype('category')


# In[26]:


# 'urlslug' feature has pattern: '\/wiki\/ *******_(****_****)' 
dc['urlslug'].head(5)


# In[27]:


# We replace 'wiki' with ''
dc['urlslug'] = dc['urlslug'].str.replace('wiki', '')


# In[28]:


# After that, we split the remaining pattern '\/\/ *******_(****_****)' with '(' and 'list1': ('\/\/ *******_', '****_****)' )  is created as new column.
dc['list1']= dc['urlslug'].str.split('(')


# In[29]:


# 'urlslug' feature is dropped.
dc.drop('urlslug', axis=1)
dc.head(2)


# In[30]:


# First element of 'list1' is assigned to 'urlslug1', second element is assigned to 'urlslug2' columns.
dc['urlslug1'] = dc['list1'].str.get(0)


# In[31]:


dc['urlslug2'] = dc['list1'].str.get(1)


# In[32]:


# 'list1' column is dropped
dc.drop('list1', axis=1,inplace=True)


# In[33]:


dc.head(5)


# In[34]:


# '_' character of 'urlslug1' and 'urlslug2' columns are replaced with ''.
dc['urlslug1'] = dc['urlslug1'].str.replace('_', ' ')


# In[35]:


dc['urlslug2'] = dc['urlslug2'].str.replace('_', ' ')


# In[36]:


# ')' character of 'urlslug2' column  is replaced with ''
dc['urlslug2'] = dc['urlslug2'].str.replace(')', '')


# In[37]:


dc.head()
#'urlslug1' still contains  '\/\/' characters.


# In[38]:


# First, with '\\', '\' character  is replaced with  ''.
dc['urlslug1'] = dc['urlslug1'].str.replace('\\', '')


# In[39]:


# With '\/', '/' character  is replaced with  ''.
dc['urlslug1'] = dc['urlslug1'].str.replace('\/', '')


# In[40]:


# We control whether there are space character ' ' in that column.
dc['urlslug1'].apply(len).head()


# In[41]:


# For example, 'batman' has 6 characters, but we can see upward that length is 7, 'supermen' has 8 characters, but we can see upward that length is 7
# We should use .strip() method in order to drop space characters.
dc['urlslug1'] = dc['urlslug1'].str.strip()


# In[42]:


# Could do it?
dc['urlslug1'].apply(len).head()
# Yes. length of 'batman' is 6, and so on..


# In[43]:


dc.head()


# In[44]:


# Drop 'urlslug':
dc.drop('urlslug', axis=1, inplace=True)


# In[45]:


# Feature 'name' is the same with 'urlslug1' + 'urlslug2' so we can drop:
dc.drop('name', axis=1, inplace=True)


# In[46]:


dc.head()


# In[47]:


#'ALIGN', 'EYE' and 'HAIR' columns are also categorical data types.
dc['ALIGN'].value_counts()


# In[48]:


dc['ALIGN'] = dc['ALIGN'].astype('category')


# In[49]:


dc['EYE'].value_counts()


# In[50]:


dc['EYE'] = dc['EYE'].astype('category')


# In[51]:


dc['HAIR'].value_counts()


# In[52]:


dc['HAIR'] = dc['HAIR'].astype('category')


# In[53]:


# 'FIRST APPEARANCE' column includes year and month pair. Also, we have 'YEAR' column. 
# We split this column
dc['list2'] = dc['FIRST APPEARANCE'].str.split(',')


# In[54]:


#'MONTH' column is created from the second element of 'list2'
dc['MONTH'] = dc['list2'].str.get(1)


# In[55]:


dc['MONTH'].value_counts()


# In[56]:


dc['MONTH'] = dc['MONTH'].str.replace('August','Aug')
dc['MONTH'] = dc['MONTH'].str.replace('December','Dec')
dc['MONTH'] = dc['MONTH'].str.replace('October','Oct')
dc['MONTH'] = dc['MONTH'].str.replace('September','Sep')
dc['MONTH'] = dc['MONTH'].str.replace('July','Jul')
dc['MONTH'] = dc['MONTH'].str.replace('February','Feb')
dc['MONTH'] = dc['MONTH'].str.replace('June','Jun')
dc['MONTH'] = dc['MONTH'].str.replace('March','Mar')
dc['MONTH'] = dc['MONTH'].str.replace('January','Jan')
dc['MONTH'] = dc['MONTH'].str.replace('April','Apr')
dc['MONTH'] = dc['MONTH'].str.replace('November','Nov')


# In[57]:


# 'MONTH' is also categorical
dc['MONTH'] = dc['MONTH'].astype('category')


# In[58]:


# No need to 'FIRST APPEARANCE' and 'list2' columns
dc.drop('FIRST APPEARANCE', axis=1, inplace=True)


# In[59]:


dc.drop('list2', axis=1, inplace=True)


# In[60]:


dc.head()


# In[61]:


dc.info()


# In[62]:


# There are only 3 numeric columns.
dc.describe()


# In[63]:


# 'page_id' column is not needed.
dc.drop('page_id', axis= 1, inplace=True)


# In[64]:


# There are NaN values, but data types are regulated, complex columns are cleaned.
dc.head()


# **MARVEL**

# In[65]:


mr.head()


# In[66]:


# Same regulations are performed to marvel data.
mr.drop(labels= ['page_id', 'urlslug', 'GSM'], axis= 1, inplace=True)


# In[67]:


mr['list1'] = mr['name'].str.split('(')


# In[68]:


mr['name1'] = mr['list1'].str.get(0)


# In[69]:


mr['name2'] = mr['list1'].str.get(1)


# In[70]:


mr.drop(['name', 'list1'], axis = 1, inplace=True)


# In[71]:


mr['name2'] = mr['name2'].str.replace(')', '')


# In[72]:


mr['name2'] = mr['name2'].str.replace('\\', '')


# In[73]:


mr['name2'] = mr['name2'].str.replace('"', '')


# In[74]:


mr['name2'] = mr['name2'].str.replace('-', '')


# In[75]:


mr['name1'] = mr['name1'].str.replace('-', '')


# In[76]:


mr['list2'] = mr['FIRST APPEARANCE'].str.split('-')


# In[77]:


mr['MONTH'] = mr['list2'].str.get(0)


# In[78]:


mr.drop(['FIRST APPEARANCE', 'list2'], axis=1, inplace=True)


# In[79]:


mr.head()


# In[80]:


mr['ID'].value_counts()


# In[81]:


mr['ID'] = mr['ID'].astype('category')


# In[82]:


mr['ALIGN'].value_counts()


# In[83]:


mr['ALIGN'] = mr['ALIGN'].astype('category')


# In[84]:


mr['EYE'].value_counts()


# In[85]:


mr['EYE'] = mr['EYE'].astype('category')


# In[86]:


mr['HAIR'].value_counts()


# In[87]:


mr['HAIR'] = mr['HAIR'].astype('category')


# In[88]:


mr['SEX'].value_counts()


# In[89]:


mr['SEX'] = mr['SEX'].astype('category')


# In[90]:


mr['ALIVE'].value_counts()


# In[91]:


mr['ALIVE'] = mr['ALIVE'].astype('category')


# In[92]:


mr['MONTH'].value_counts()


# In[93]:


mr['MONTH'] = mr['MONTH'].astype('category')


# In[94]:


mr.head()


# **DC AND MARVEL**

# In[95]:


mr.head(2)


# In[96]:


dc.head(2)


# In[97]:


# Column names should be identical:
dc.rename(columns={'urlslug1': 'name1',
                   'urlslug2': 'name2',
                  'YEAR': 'Year'}, inplace=True)


# In[98]:


dc.head(2)


# In[ ]:




