#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# Any results you write to the current directory are saved as output.


# In[ ]:


import glob2
p1=glob2.glob('/kaggle/input/*/*.csv')


# In[ ]:


#p1.head()


# In[ ]:


df1=pd.read_csv('/kaggle/input/kernel6e40bb3926/submission1.csv')
df2a=pd.read_csv('/kaggle/input/part2a/submission2a.csv')
df2b=pd.read_csv('/kaggle/input/part2b/submission2b.csv')
df3a=pd.read_csv('/kaggle/input/part3a/submission1.csv')
df3b=pd.read_csv('/kaggle/input/part3b/submission3b.csv')
df4=pd.read_csv('/kaggle/input/part4/submission4.csv')


# In[ ]:


df1.head()


# In[ ]:


df2a.head()


# In[ ]:


df2b.head()


# In[ ]:


df3a.head()


# In[ ]:


df3b.head()


# In[ ]:


df4.head()


# In[ ]:


df_final=pd.concat([df1, df2a, df2b, df3a, df3b, df4])
df_final.head()


# In[ ]:


df_final.shape


# In[ ]:


df_final.to_csv('submission.csv',index=None)


# In[ ]:


get_ipython().system('ls')

