#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


jg=pd.read_excel('/kaggle/input/junglee-mukeshk/101 Pool Game Case Study - Data Set_withAnalysis.xlsx','Sheet_1_crosstab (1)')
print(jg)


# In[ ]:


jg.head()


# In[ ]:


jg.info()


# In[ ]:


jg.columns


# In[ ]:


del(jg[143546424.25])


# In[ ]:


jg.columns


# In[ ]:


del(jg['Unnamed: 11'])


# In[ ]:


jg.columns


# In[ ]:


jg.describe()


# In[ ]:


len(jg['Date'].unique())


# In[ ]:


len(jg['Entry Fee'].unique())


# In[ ]:


len(jg['Entry Fee'])


# In[ ]:


len(jg['Cut %'])


# In[ ]:


len(jg['Cut %'].unique())


# In[ ]:




