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


airQ_df = pd.read_csv("/kaggle/input/datasetucimlairquality/AirQualityUCI.csv")
airQ_df.head()


# In[ ]:


airQ_df.columns


# In[ ]:


airQ_df['DateTime'] = airQ_df.Date + ' '+airQ_df.Time
airQ_df['DateTime'] = pd.to_datetime(airQ_df.DateTime)
airQ_df.head()


# In[ ]:


airQ_df.dtypes


# In[ ]:


airQ_df.shape


# In[ ]:


airQ_df.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.matshow(airQ_df.corr())
plt.colorbar()
plt.show()


# In[ ]:


plt.scatter(airQ_df['DateTime'], airQ_df['CO_GT'])
plt.show()

