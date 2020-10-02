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


sales_train=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sales_train.head()


# In[ ]:


#dept_id unique elements
sales_train['dept_id'].unique()


# In[ ]:


sales_train.groupby('state_id').mean()


# In[ ]:


sales_train.groupby('item_id').mean()


# In[ ]:


sales_train.groupby(['state_id','item_id']).mean()


# In[ ]:





# In[ ]:


sales_train.describe()


# In[ ]:


sales_train.T


# In[ ]:


sales_train.isna().sum()


# In[ ]:


calendar=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calendar.head()


# In[ ]:


calendar.describe()


# In[ ]:


calendar.isna().sum()


# In[ ]:


sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sell_prices.head()


# In[ ]:


sell_prices.shape,calendar.shape,sales_train.shape


# In[ ]:


sell_prices.store_id.value_counts()


# In[ ]:


#group by store_id
sell_prices.groupby('store_id').mean()


# In[ ]:


sell_prices.describe()


# In[ ]:


sell_prices.isna().sum()


# In[ ]:


sell_prices.hist(column='sell_price',by='store_id',figsize=(13,17))

