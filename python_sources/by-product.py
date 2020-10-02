#!/usr/bin/env python
# coding: utf-8

# In this kernel, I will try to explore product category as per discussion :https://www.kaggle.com/c/ieee-fraud-detection/discussion/108467#latest-629718 (Laevatein, Chris Deotte and snovik) 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from plotnine import *
import warnings
warnings.filterwarnings('ignore') 


# In[ ]:


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


# In[ ]:


train_transaction[['ProductCD','isFraud']] = train_transaction[['ProductCD','isFraud']].astype('category')


# In[ ]:


#train_transaction['cents'] = train_transaction.TransactionAmt.astype('str').str.split('.').str[1].astype('int8')
train_transaction['cents'] = np.round( train_transaction['TransactionAmt'] - np.floor(train_transaction['TransactionAmt']),2 )


# In[ ]:


train_transaction['cents'].describe()


# In[ ]:


ggplot(train_transaction, aes(x='TransactionDT',y='cents', color='isFraud'))+geom_point(size=0.1, alpha=0.3) +facet_wrap( facets='ProductCD', nrow=5) +theme(figure_size=(12, 20))


# In[ ]:


train_transaction.groupby('ProductCD')[['dist1','dist2']].count()


# In[ ]:


df = train_transaction.groupby('ProductCD').count().reset_index()
df = df.set_index('ProductCD').stack().reset_index(name='Value')
df['Value'] = df['Value'] / df['ProductCD'].map(train_transaction['ProductCD'].value_counts()).astype('int')


# In[ ]:


ggplot(df, aes(x='level_1',y='Value'))+geom_bar(stat='identity')+facet_wrap( facets='ProductCD', nrow=5) +theme(figure_size=(12, 20))


# In[ ]:




