#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


MainDir = '/kaggle/input/m5-forecasting-accuracy/'


# In[ ]:


Calander_df = pd.read_csv(MainDir + 'calendar.csv')
Calander_df


# In[ ]:


Calander_df.columns


# In[ ]:


Calander_df.info()


# In[ ]:


sales_df = pd.read_csv(MainDir + 'sell_prices.csv')
sales_df


# In[ ]:


sales_df.loc[:,['store_id','item_id','sell_price']].groupby(['store_id','item_id']).sell_price.sum()


# In[ ]:


sales_train_df = pd.read_csv(MainDir + 'sales_train_validation.csv')
sales_train_df


# In[ ]:


sales_train_df.iloc[:,4:].groupby(['store_id','state_id']).sum()


# In[ ]:


sales_train_df.columns


# In[ ]:


sales_train_df.iloc[:,0:6]


# In[ ]:


sales_train_df.item_id.value_counts()


# In[ ]:


sales_train_df.state_id.value_counts()


# In[ ]:


Sub_df = pd.read_csv(MainDir + 'sample_submission.csv')


# In[ ]:


Sub_df.columns


# In[ ]:


Sub_df


# In[ ]:




