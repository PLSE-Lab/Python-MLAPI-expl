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
print('Printing Only Few Files..')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# registerting Helper Module Path
import sys
sys.path.append('/kaggle/input/stock-market-india')


# ## Helper Module

# In[ ]:


import StockMarketHelper as helper


# ### Master File
# Loading master file. It contains key and addtional information of the stock price avaiable in dataset.
# 
#     Parameter:
# 
#     use_hdf: use hdf file to load data (default True)

# In[ ]:


helper.read_master()


# ## Filter symbol Function
# Helper function to filter individual stock data.
# 
#     Parameter:
# 
#     query: Similar to pandas query input except here if it doesn't contain =, < or > then 
#             query will be replaced by f'tradingsymbol == {query}" or name == "{query}"
#     
#     search: It will filter all the symbols with contain '@search' in tradingsymbol or name.
#     
#     return_all: If true return all matches else it will ask user to enter row number (default False)
#     
#     return_empty: If true return empty matches else raise error (default False)
#     
#     use_hdf: use hdf file to load data (default True)

# In[ ]:


# searching stock
helper.filter_symbol(search='NIFTY', return_all=True)


# In[ ]:


helper.filter_symbol(search='NIFTY', return_all=False)


# ## read_from_key Function
# It will return stock market data attached to a input key, with additional parameter to use_hdf data.
# 
#     Parameter:
# 
#     key: key should be avaialble in master file
#     
#     use_hdf: use hdf file to load data (default True)

# In[ ]:


helper.read_from_key('NIFTY_50__EQ__INDICES__NSE__MINUTE')


# ## Read Data Function
# Helper function to read individual stock data.
# 
# It will first call filter symbol with return_all and return_empty to false. And later call read_from_key.
# 
#     Parameter:
# 
#     query: Similar to pandas query input except here if it doesn't contain =, < or > then 
#             query will be replaced by f'tradingsymbol == {query}" or name == "{query}"
#     
#     search: It will filter all the symbols with contain '@search' in tradingsymbol or name.
#     
#     return_all: If true return all matches else it will ask user to enter row number (default False)
#     
#     return_empty: If true return empty matches else raise error (default False)
#     
#     use_hdf: use hdf file to load data (default True)
# 
# 
#     This Function will call filter_symbol with return_all=False, return_empty=False, which mean
#       - it will ask user to enter row number if mutiple symbol satisfies the criteria
#       - and it will raise error if no symbol satisfies the criteria.

# In[ ]:


helper.read_data(search='AI')


# In[ ]:


helper.read_data(query='BHARTIARTL')


# In[ ]:




