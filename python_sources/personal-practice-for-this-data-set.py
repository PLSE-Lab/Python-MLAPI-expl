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
sal=pd.read_csv('../input/Salaries.csv')


# In[ ]:


sal.head()


# In[ ]:


sal.info()


# In[ ]:


sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']


# In[ ]:


sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']


# In[ ]:


sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].max()]
sal.loc[sal['TotalPayBenefits'].idxmax()]


# In[ ]:


sal.iloc[sal['TotalPayBenefits'].idxmin()]


# In[ ]:


sal.groupby('Year').mean()['BasePay']


# In[ ]:


sal['JobTitle'].nunique()


# In[ ]:


sal['JobTitle'].value_counts().head(5)


# In[ ]:


sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1)


# In[ ]:


def chief_string (title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False


# In[ ]:


sal['JobTitle'].apply(lambda x: chief_string(x))


# In[ ]:




