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


data=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head(8)


# In[ ]:


data.tail(8)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data["PaymentMethod"]


# In[ ]:


data.drop(["SeniorCitizen"],axis=1,inplace=True)
data.head()


# In[ ]:


data.gender=[1 if each=="Male" else 0 for each in data.gender]


# In[ ]:


data.gender.value_counts()


# In[ ]:


data.info()


# In[ ]:


data.PaymentMethod.value_counts()

