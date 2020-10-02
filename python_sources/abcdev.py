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


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot


# In[ ]:


pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame
pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame


# In[ ]:


df_onsels = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
df_offsels = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')
df_product = pd.read_csv('/kaggle/input/uisummerschool/Product.csv')
df_marketing = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')


# In[ ]:


df_onsels.head()


# In[ ]:


df_onsels['Date'] = df_onsels['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
df_onsels.head()


# In[ ]:


grouped = df_onsels.groupby(['Date'])['Revenue'].sum().reset_index(name='total_revenue')
grouped.head()


# In[ ]:


grouped1 = df_onsels.groupby(['Date'])['Quantity'].sum().reset_index(name='Daily_quantity')
grouped1.head()


# In[ ]:


df_marketing.head()


# In[ ]:


df_marketing['Date'] = pd.to_datetime(df_marketing['Date'])

df_marketing


# In[ ]:


grouped['Online Spend'] = df_marketing['Online Spend']
grouped


# In[ ]:


import matplotlib.pyplot as plt

plt.plot( 'Date', 'Online Spend', data=grouped[grouped['Date'] < '2017-02-01'], color='red')
plt.plot( 'Date', 'total_revenue', data=grouped[grouped['Date'] < '2017-02-01'], color='blue')
plt.show()


# In[ ]:


grouped1['Online Spend'] = df_marketing['Online Spend']
grouped1


# In[ ]:


plt.plot( 'Date', 'Online Spend', data=grouped1[grouped1['Date'] > '2017-11-01'], color='red')
plt.plot( 'Date', 'Daily_quantity', data=grouped1[grouped1['Date'] > '2017-11-01'], color='blue')
plt.show()

