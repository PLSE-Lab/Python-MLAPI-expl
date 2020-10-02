#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data1 = pd.read_csv('../input/merchants.csv')
data2 = pd.read_csv('../input/new_merchant_transactions.csv')
data3 = pd.read_csv('../input/test.csv')
data4 = pd.read_csv('../input/train.csv')
data5 = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


data1.info()


# In[ ]:


data2.info()


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,center = 0,vmax = 1,vmin = -0.2)
plt.show()


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data2.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,center = 0,vmax = 1,vmin = -0.2)
plt.show()


# In[ ]:


data1.head()


# In[ ]:


data1.tail()


# In[ ]:


data2.head()


# In[ ]:


data2.tail()


# In[ ]:


data1.plot(kind='scatter', x='avg_sales_lag3', y='avg_sales_lag6',alpha = 0.9,color = 'green',figsize=(16,16))
plt.show()


# In[ ]:


data1.plot(kind='scatter', x='avg_sales_lag3', y='avg_sales_lag12',alpha = 0.9,color = 'red',figsize=(16,16))
plt.show()


# In[ ]:


data1.plot(kind='scatter', x='avg_sales_lag6', y='avg_sales_lag12',alpha = 0.9,color = 'blue',figsize=(16,16))
plt.show()


# In[ ]:


data2.columns


# In[ ]:


data2.plot(kind='scatter', x='purchase_amount', y='installments',alpha = 0.9,color = 'orange',figsize=(16,16))
plt.show()


# In[ ]:


#data2.plot(kind='scatter', x='card_id', y='city_id',alpha = 0.9,color = 'orange',figsize=(16,16))
#plt.show()


# In[ ]:


data1.info()


# In[ ]:


print(data1['merchant_id'].value_counts(dropna =False))


# In[ ]:


print(data1['category_4'].value_counts(dropna =False))


# In[ ]:


data2.info()


# In[ ]:


print(data2['authorized_flag'].value_counts(dropna =False))


# In[ ]:


print(data2['city_id'].value_counts(dropna =False))


# In[ ]:


print(data2['card_id'].value_counts(dropna =False))


# In[ ]:


print(data2['category_3'].value_counts(dropna =False))


# In[ ]:


data1.describe()


# In[ ]:


data2.describe()


# In[ ]:


data2.boxplot(column='purchase_amount')


# In[ ]:


data_new1 = data1.head()
data_new1


# In[ ]:


melted1 = pd.melt(frame = data_new1,id_vars = 'merchant_id',value_vars = ['merchant_category_id','subsector_id'])
melted1


# In[ ]:


data_new2 = data2.head()
data_new2


# In[ ]:


melted2 = pd.melt(frame = data_new2,id_vars = 'card_id',value_vars = ['purchase_amount','installments'])
melted2


# In[ ]:


# Reverse of melting (pivoting_data)
melted1.pivot(index = 'merchant_id',columns = 'variable',values = 'value')


# In[ ]:


# concetenating data 
conc_data = data1.head()
conc_data1 = data1.tail()
conc_data_row = pd.concat([conc_data,conc_data1],axis = 0,ignore_index = False)
conc_data_row


# In[ ]:


data_0 = data1['merchant_category_id'].head()
data_1 = data1['subsector_id'].head()
conc_data_col = pd.concat([data_0,data_1],axis = 1,ignore_index = False)
conc_data_col


# In[ ]:


data1['avg_sales_lag3'].fillna('empty',inplace = True)


# In[ ]:


data1.head()


# In[ ]:


data1['avg_sales_lag6'].fillna(0.0,inplace = True)


# In[ ]:


assert data1['avg_sales_lag6'].notnull().all()


# In[ ]:


data1['avg_sales_lag6'] = data1['avg_sales_lag6'].astype('int64')


# In[ ]:


data1.info()


# In[ ]:


data1.columns


# In[ ]:


data2.columns


# In[ ]:


data1['category_1'] = data1['category_1'].astype("category")

