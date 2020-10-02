#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


f19 = pd.read_csv('/kaggle/input/fifa19/data.csv')


# In[ ]:


f19.head()


# In[ ]:


f19.columns


# In[ ]:


atlutd = f19[f19.Club == 'Atlanta United']
atlutd.head()


# In[ ]:


us = f19[f19.Nationality == 'United States']
us.Club.unique()  # Clubs that have American players


# In[ ]:


sns.distplot(f19['Age']);


# In[ ]:


sns.distplot(us['Age'])


# In[ ]:


sns.distplot(atlutd['Age'])


# In[ ]:




