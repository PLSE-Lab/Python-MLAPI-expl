#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# r_dataisbeautiful_posts.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')


# In[ ]:


df1.head(5)


# In[ ]:


df1.drop(['total_awards_received','full_link','removed_by','created_utc','awarders'],axis=1)


# In[ ]:


df1.count()


# In[ ]:


df1.max()


# In[ ]:


df1.isnull()


# In[ ]:


df1.score.plot()


# In[ ]:




