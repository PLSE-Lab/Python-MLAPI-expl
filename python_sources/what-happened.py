#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from scipy.signal import savgol_filter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train.head()


# In[ ]:


train.shape


# In[ ]:


plt.figure(figsize=(15,15))
_ = plt.plot(savgol_filter(train.isFraud, 10001, 1,0))


# In[ ]:


plt.figure(figsize=(15,15))
_ = plt.plot(savgol_filter(train.TransactionDT, 10001, 1,1))


# The first 120000 records are much lower than the rest
