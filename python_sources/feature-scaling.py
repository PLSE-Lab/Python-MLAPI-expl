#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
from sklearn import preprocessing


# In[ ]:


#example of trainning data
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])


# In[ ]:


X_scaled = preprocessing.scale(X)
X_scaled


# In[ ]:


X_scaled.mean(axis = 0)


# In[ ]:


X_scaled.std(axis= 0)


# In[ ]:


scaler = preprocessing.StandardScaler()
scaler.fit(X)


# In[ ]:


X_test = [[1,1,0]]
scaler.transform(X_test)


# In[ ]:




