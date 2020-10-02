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


import matplotlib as plt


# In[ ]:


data_sales_online = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
data_sales_offline = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')


# In[ ]:


data_sales_online


# In[ ]:


data_sales_online.describe()


# In[ ]:


data_sales_online.info()


# In[ ]:


data_sales_online.plot(kind = 'line',x = 'Date', y = 'Quantity' , color = 'red')
data_sales_offline.plot(kind = 'line', x ='InvoiceDate', y = 'Quantity', color = 'blue'  )


# Insight : In this red chart shows online sales quantities in a year,blue shows offline sales quantities. 

# 

# In[ ]:


#join.plot(x = 'Date', y = 'Quantity')


# In[ ]:




