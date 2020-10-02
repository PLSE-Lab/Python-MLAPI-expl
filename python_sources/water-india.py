#!/usr/bin/env python
# coding: utf-8

# My first kernel. It is still in progress.

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


data1 = pd.read_csv('../input/chennai_reservoir_levels.csv')
data1.head()


# In[ ]:


import seaborn as sns
sns.pairplot(data1)


# In[ ]:




