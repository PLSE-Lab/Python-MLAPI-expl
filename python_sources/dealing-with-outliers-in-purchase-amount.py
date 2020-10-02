#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


history = pd.read_csv("../input/historical_transactions.csv", low_memory=True)


# In[ ]:


history.shape


# In[ ]:


history['purchase_amount'].hist(bins=100)


# As you can see we have outliers in this distribution we can confirm this conclusion by drawing the box plot 

# In[ ]:


history['purchase_amount'].plot.box()


# we have point ( transaction) with purchase amount higer than 6000000. we try to eliminate it 

# In[ ]:


history=history[history['purchase_amount']<5000000]


# In[ ]:


history['purchase_amount'].hist()


# After elimination of that point, we still have outliers in this feature so, in the next few cells, I drew some histograms in order to allocate outliers, fill free to skip them and focus on the block final bolck  when I used **pd.cut **

# In[ ]:


history[history['purchase_amount']>25000]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<20000]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<2500]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<500]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<100]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<10]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<1]['purchase_amount'].hist()


# In[ ]:


history[history['purchase_amount']<1].shape


# In[ ]:


history[history['purchase_amount']<1].shape[0]/history.shape[0]


# In[ ]:


history[history['purchase_amount']>1]['purchase_amount'].hist()


# In[ ]:


history[(history['purchase_amount']>1) & (history['purchase_amount']<10)]['purchase_amount'].hist()


# In[ ]:


history[(history['purchase_amount']>10) & (history['purchase_amount']<100)]['purchase_amount'].hist()


# In[ ]:


history[(history['purchase_amount']>100) & (history['purchase_amount']<1000)]['purchase_amount'].hist()


# In[ ]:


history[(history['purchase_amount']>10000) & (history['purchase_amount']<100000)]['purchase_amount'].hist()


# In[ ]:


bins = [-1,1,10,100,1000,10000,100000,1000000,10000000]
history['binned_purchase_amount'] = pd.cut(history['purchase_amount'], bins)


# In[ ]:


binned_purchase_amount_cnt = history.groupby("binned_purchase_amount")['binned_purchase_amount'].count().reset_index(name='binned_purchase_amount_cnt')
binned_purchase_amount_cnt.columns = ['purchase_amount','binned_purchase_amount_cnt']


# In[ ]:


binned_purchase_amount_cnt['percent']=binned_purchase_amount_cnt['binned_purchase_amount_cnt']*100/binned_purchase_amount_cnt['binned_purchase_amount_cnt'].sum()


# In[ ]:


binned_purchase_amount_cnt


# As you can see more than 98% of this distribution still between -1 and 1, we define data outside this interval as  outliers.

# In[ ]:


binned_purchase_amount_cnt.plot.bar(y='percent', x='purchase_amount')


# In order to reduce variance and keep the sign of the distribution, I used this transformation : sign(x)*ln(1+abs(x))

# In[ ]:


def log_1(x):
    return math.copysign(1,x)*math.log(1+abs(x))


# In[ ]:


history['log_purchase_amount']=history['purchase_amount'].apply(lambda x :log_1(x) )


# In[ ]:


history['log_purchase_amount'].hist()


# instead of eliminating outliers, it may be useful to score the loyalty of the customer, I create a new feature to distinguish between normal data and outliers.

# In[ ]:


history['purchase_amount_outliers'] = 0
history.loc[history['purchase_amount'] >1, 'purchase_amount_outliers'] = 1


# In[ ]:




