#!/usr/bin/env python
# coding: utf-8

# Hi To you all this is my first attempt on the Dataset 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


banks = pd.read_csv('../input/banks.csv')
banks.info()


# In[ ]:


banks.head()


# In[ ]:


banks.isnull().sum()


# In[ ]:


corr = banks[banks.columns].corr()
sns.heatmap(corr,annot = True)
##as you can see on the corr plot Total Deposits and Total Assets have a  strong positive correlation 


# In[ ]:


banks.get('Institution Type').unique()#these are categorical features which i will transform later


# In[ ]:



banks.get('Transaction Type').unique()#these are categorical features which i will transform later


# In[ ]:



banks.get('Charter Type').unique()#'STATE', 'FEDERAL', 'FEDERAL/STATE#these are categorical features which i will transform later


# In[ ]:



banks.get('Insurance Fund').unique()#these are categorical features which i will transform later


# In[ ]:


banks['liquidity'] = banks['Total Deposits']/banks['Total Assets'] * 100


# In[ ]:


banks.plot.scatter(x = 'Total Assets', y = 'Estimated Loss (2015)')
banks.plot.scatter(x = 'Total Deposits', y = 'Estimated Loss (2015)')
banks.plot.scatter(x = 'Total Assets', y = 'Total Deposits')
banks.plot.scatter(x = 'liquidity', y = 'Estimated Loss (2015)')


# In[ ]:


sns.stripplot(x = 'Charter Type', y = 'Estimated Loss (2015)', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Charter Type', y = 'Total Assets', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Charter Type', y = 'Total Deposits', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Institution Type', y = 'Estimated Loss (2015)', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Insurance Fund', y = 'Estimated Loss (2015)', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Insurance Fund', y = 'Total Assets', data = banks, jitter = True);


# In[ ]:


sns.countplot( y = 'Institution Type', data = banks);


# In[ ]:


sns.stripplot(x = 'Transaction Type', y = 'Estimated Loss (2015)', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Transaction Type', y = 'Total Assets', data = banks, jitter = True);


# In[ ]:


sns.stripplot(x = 'Transaction Type', y = 'Total Deposits', data = banks, jitter = True);


# In[ ]:


banks['liquidity'] = banks['Total Deposits']/banks['Total Assets'] * 100
banks['liquidity'].hist(bins = 100)


# In[ ]:


sns.distplot(liquidity,hist = True, bins = 100)


# In[ ]:


HQ = banks['Headquarters'].value_counts()
HQ.head(20)                               # cities  Where most Headquarters of banks that failed where located


# In[ ]:


HQ[:10].plot(kind='bar') 

