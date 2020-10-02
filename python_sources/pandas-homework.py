#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[31]:


import pandas as pd
import pandas_datareader.data as web

start_date = '2018-01-01'
end_date = '2018-09-30'

stocks = ['AAC', 'AAAU', 'AAL', 'AAPL']
data_stocks = web.DataReader(stocks, 'yahoo', start_date, end_date)
data_stocks.head()


# In[32]:


data_stocks.columns


# In[33]:


data_stocks = data_stocks.stack(['Symbols']).reset_index()


# In[34]:


data_stocks.groupby('Symbols').agg(['median', 'mean', 'sum'])


# In[12]:


data_stocks.groupby(['Symbols']).describe()


# In[35]:


data_stocks.pivot_table(index = ['Symbols'], aggfunc = 'mean')


# In[29]:


data_stocks.head()


# In[39]:


# Tinh loi sua theo gia dong cua
import numpy as np
data_return = pd.DataFrame({'Date': data_stocks['Date'].unique()})
data_return = data_return.set_index(['Date'])
for col in stocks:
    rname = 'R_'+col
    data_stock = data_stocks[data_stocks['Symbols'] == col]
    data_stock[rname] = np.log(data_stocks['Close']) - np.log(data_stocks['Close'].shift(1))
    data_stock = data_stock[['Date', rname]].set_index('Date')
    data_return = data_return.join(data_stock, how = 'left')

data_return.head()

