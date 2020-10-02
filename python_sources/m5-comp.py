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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


p = pd.read_csv('/kaggle/input/m5-first-public-notebook-under-0-50/submission.csv')
t = pd.read_csv('/kaggle/input/m5-accuracy-tweedie-is-back/submission.csv')


# In[ ]:


p.head()


# In[ ]:


t.head()


# In[ ]:


p.corrwith(t, axis = 0)


# In[ ]:


p = p.sort_values(by = 'id').reset_index(drop = True)
t = t.sort_values(by = 'id').reset_index(drop = True)
sub = p.copy()

for i in sub.columns :
    if i != 'id' :
        sub[i] = 0.3*p[i] + 0.2*t[i]
        
sub.to_csv('sub.csv', index = False)

