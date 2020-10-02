#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


sns.pairplot(train_data)


# In[ ]:


train_data.columns


# In[ ]:


dataset = train_data.pivot_table(index = ['shop_id','item_id'],columns = ['date_block_num'],values = ['item_cnt_day'],fill_value = 0)


# In[ ]:


dataset.reset_index(inplace = True)


# In[ ]:


dataset = pd.merge(test_data,dataset,how = 'left',on = ['shop_id','item_id'])


# In[ ]:


dataset.head()


# In[ ]:


dataset.fillna(0,inplace = True)
dataset.head()


# In[ ]:


submission_pfs = dataset.iloc[:,36]


# In[ ]:


submission_pfs.clip(0,20,inplace = True)
submission_pfs = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})


# In[ ]:


submission_pfs.head()


# In[ ]:


submission_pfs.to_csv('pfs.csv',index = False)


# In[ ]:


g = pd.read_csv('pfs.csv')


# In[ ]:


g.head()


# In[ ]:




