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
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Mencari segment mana yang spend nya paling banyak

# In[ ]:


df = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')
aa =df.drop(columns=['Date']).sum()
bb = pd.DataFrame(aa).transpose()
bb.plot(kind='bar')

# marketing offline lebih banyak spendnya


# In[ ]:


dfOffline = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')
aa =dfOffline.drop(columns=['InvoiceNo'])
bb = aa.drop(columns=['InvoiceDate'])
cc =  bb.drop(columns=['StockCode'])
dd = pd.DataFrame(cc).sum().transpose()
dd.plot(kind='bar')
# jumlah quantity offline sales yang telah terjual





# In[ ]:


dfOnline = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv', usecols=[5], names = ["b"],)
# dfOnline =dfOnline.sum()
# dfOnline.sum(axis = 0, skipna = True)
dfOnline.sum().
# aa =pd.DataFrame(dfOnline).sum().transpose()
# aa =dfOffline.drop(columns=['Delivery'])
# bb = aa.drop(columns=['Date'])
# bb
# cc =  bb.drop(columns=['Product SKU'])
# dd =  cc.drop(columns=['Product'])
# ee =  dd.drop(columns=['Avg. Price'])
# ff =  ee.drop(columns=['Revenue'])
# gg =  ff.drop(columns=['Revenue'])
# hh =  gg.drop(columns=['Tax'])
# ii =  hh.drop(columns=['Delivery'])
# jj = ii.DataFrame(cc).sum().transpose()
# ii
# ii.plot(kind='bar')

