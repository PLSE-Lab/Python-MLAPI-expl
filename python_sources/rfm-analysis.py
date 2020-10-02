#!/usr/bin/env python
# coding: utf-8

# The EDA tutorial is here : [(https://www.kaggle.com/ivy1219/customer-behaviour-observation)]
# ## In this chapter, we have two things to do:
# 1.  Classify the churned and remained user;
# 2. Build  RFM model to find the best user group and classify users.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


prior = pd.read_csv("../input/order_products__prior.csv")
orders = pd.read_csv("../input/orders.csv")
# It takes some time, so you can make a cup of tea


# In[ ]:


print(prior.columns, orders.columns)  # I have a bad memory, just need to find out the common column


# In[ ]:


alls = pd.merge(orders,prior, on=['order_id','order_id'])


# In[ ]:


alls['Total'] = alls['order_number'] * alls['add_to_cart_order']
cols = ['eval_set', 'order_dow',
       'order_hour_of_day','product_id','add_to_cart_order','order_number']
alls.drop(cols,axis =1)
alls.head()
users_churn = alls[alls.reordered == 0]  # churn is churn, churned users we have different ways to deal with
user_remain = alls[alls.reordered != 0]  # just focus on the remained users 


# In[ ]:


print(user_remain.user_id.nunique())
print(len(user_remain)/len(alls)*100)  # 58.9% user remains, not bad.


# In[ ]:


RFMtable = user_remain.rename(columns = {'Total': 'Monetary',
                  'days_since_prior_order': 'Recency',
                 'reordered': 'Frequency'},inplace = True)

#RFMtable.to_csv('RFMtable.csv') 
# sometimes the kernel died without any warning and I don't want to load the big file again and again, so just in case


# In[ ]:


RFMtable = user_remain.groupby('user_id').agg({'Recency': lambda x: x.max(),
                                              'Frequency': lambda x: len(x),
                                              'Monetary': lambda x: x.sum()})
RFMtable.head(2)


# In[ ]:


quantiles = RFMtable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[ ]:


## create the RFM segmentation table
RFMSegmentation = RFMtable
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[ ]:


RFMSegmentation['R_Quantile'] = RFMSegmentation['Recency'].apply(RClass, args=('Recency',quantiles,))
RFMSegmentation['F_Quantile'] = RFMSegmentation['Frequency'].apply(FMClass, args=('Frequency',quantiles,))
RFMSegmentation['M_Quantile'] = RFMSegmentation['Monetary'].apply(FMClass, args=('Monetary',quantiles,))


# In[ ]:


RFMSegmentation['RFMClass'] = RFMSegmentation.R_Quantile.map(str)                             + RFMSegmentation.F_Quantile.map(str)                             + RFMSegmentation.M_Quantile.map(str)
RFMSegmentation.head()


# In[ ]:


sns.set_palette('Paired')
sns.set(rc={'image.cmap': 'coolwarm'})

fig,axes = plt.subplots(3,1,figsize = (16,8))
sns.countplot(y = RFMSegmentation.R_Quantile,ax = axes[0])
sns.countplot(y = RFMSegmentation.F_Quantile,ax = axes[1])
sns.countplot(y = RFMSegmentation.M_Quantile,ax = axes[2])


# 1. The number of higher recency users is fewer than that of lower recency user, which is acceptable and make sense .
# 2. The number of lower frequency users is the most,  acceptable and make sense . Other three clusters are quite similar, it shows the supermarket managed its customer very well.
# 3. From the Monetary value distribution, we know the dataset has been'modified' a little. It can't be true in real world data.

# In[ ]:


RFMSegmentation.RFMClass.value_counts()
# see! not bad


# In[ ]:


# let us see our best users
RFMSegmentation[RFMSegmentation['RFMClass']=='111'].sort_values('Monetary', ascending=False).head(5)


# In[ ]:


# let us see also good loyalty but spend less money users
RFMSegmentation[RFMSegmentation['RFMClass']=='134'].sort_values('Monetary', ascending= True).head(5)


# 
# Both groups of user are high loyalty cuz the R_quantile is 1, which means they always come back and buy somthing in a not quite long period.
# 
# However, loyalty does not means money or profit. We can see clearly that the Monetary value of the two groups are quite different!
# 
# The more you know your customers, the better you can serve them .

# If that helps you ,please upvote me and feel free to comment!

# 
