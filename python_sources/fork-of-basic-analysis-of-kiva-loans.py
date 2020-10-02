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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/kiva_loans.csv')


# In[ ]:


df.head(20)


# In[ ]:


df.describe()


# In[ ]:


df.info


# In[ ]:


df.columns


# In[ ]:


x=df[['funded_amount', 'activity', 'sector', 'country', 'region', 'currency', 'disbursed_time', 'funded_time', 'term_in_months',
       'lender_count', 'tags', 'borrower_genders', 'repayment_interval']]


# In[ ]:


y=df['loan_amount']


# In[ ]:


sns.pairplot(x)


# In[ ]:


df.columns


# In[ ]:


sns.distplot(df['loan_amount'],kde=False,bins=1000)


# In[ ]:


sns.countplot(x='sector',data=df)


# In[ ]:





# In[ ]:


df.groupby('sector')


# In[ ]:


g=df.groupby('sector')


# In[ ]:


g.describe()


# In[ ]:


g.describe().plot(kind='bar')


# In[ ]:


for sector,sector_df in g:
    print(sector)
    print(sector_df)


# In[ ]:


df.info


# In[ ]:




