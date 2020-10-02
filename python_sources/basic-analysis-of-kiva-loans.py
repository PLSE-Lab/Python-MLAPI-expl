#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('../input/kiva_loans.csv')


# In[5]:


df.head(20)


# In[6]:


df.describe()


# In[7]:


df.info


# In[8]:


df.columns


# In[11]:


x=df[['funded_amount', 'activity', 'sector', 'country', 'region', 'currency', 'disbursed_time', 'funded_time', 'term_in_months',
       'lender_count', 'tags', 'borrower_genders', 'repayment_interval']]


# In[9]:


y=df['loan_amount']


# In[12]:


sns.pairplot(x)


# In[13]:


df.columns


# In[14]:


sns.distplot(df['loan_amount'],kde=False,bins=1000)


# In[15]:


sns.countplot(x='sector',data=df)


# In[ ]:





# In[16]:


df.groupby('sector')


# In[17]:


g=df.groupby('sector')


# In[18]:


g.describe()


# In[19]:


g.describe().plot(kind='bar')


# In[20]:


for sector,sector_df in g:
    print(sector)
    print(sector_df)


# In[ ]:


df.info


# In[ ]:




