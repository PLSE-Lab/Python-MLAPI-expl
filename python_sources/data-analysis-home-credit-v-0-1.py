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


bureau_bal=pd.read_csv("../input/bureau_balance.csv")


# In[ ]:


bureau_bal


# In[ ]:


bureau_bal.shape


# In[ ]:


bureau_bal.columns


# In[ ]:


bureau_bal['SK_ID_BUREAU'].dtype


# In[ ]:


df2=bureau_bal[['SK_ID_BUREAU','MONTHS_BALANCE']]


# In[ ]:


type(df2)


# In[ ]:


type(bureau_bal['SK_ID_BUREAU'])


# In[ ]:


bureau_bal.SK_ID_BUREAU.value_counts()


# Head of the dataset

# In[ ]:


bureau_bal.head()


# In[ ]:


bureau_bal.tail()


# In[ ]:


df3=bureau_bal['SK_ID_BUREAU']==5041336


# In[ ]:


df3


# In[ ]:


pd.__version__


# In[ ]:


bureau_bal.loc[5041336]


# In[ ]:


bureau_bal.iloc[5041336]


# In[ ]:


bureau_bal.iloc[-1]


# In[ ]:


bureau_bal.loc[0:5,'SK_ID_BUREAU']


# In[ ]:


dfg=bureau_bal.groupby('SK_ID_BUREAU')


# In[ ]:


dfg


# In[ ]:


type(dfg)


# In[ ]:


dfg.sum()


# In[ ]:





# In[ ]:




