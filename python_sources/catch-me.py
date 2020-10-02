#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df1 = pd.read_csv('../input/blendpublic/submission (2).csv')
df2 = pd.read_csv('../input/blendpublic/submission_Yury_Kashnitskiy.csv')
df1['target'] = 0.8*df1['target'] +  0.3*df2['target']
df1.head()


# In[ ]:


df1.to_csv('sub.csv',index=False)

