#!/usr/bin/env python
# coding: utf-8

# These features are inspired from raddar findings and suggestions: https://www.kaggle.com/raddar/target-true-meaning-revealed
# 
# It helped me improve my CV and LB by 0.002; hope it helps you too.
# I have taken two months gap since it is given in data dictionary that target is "Loyalty numerical score calculated 2 months after historical and evaluation period"

# In[ ]:


import pandas as pd
import numpy as np
import datetime

import gc
import sys
import os


# In[ ]:


### 
# train = pd.read_csv('../input/train.csv')
hist = pd.read_csv('../input/historical_transactions.csv')
# new = pd.read_csv('../input/new_merchant_transactions.csv')
# merchant = pd.read_csv('../input/merchants.csv')


# In[ ]:


agg_func = {
        'purchase_amount': ['sum'],
        'installments':['sum']
        }
    
agg_history = hist.groupby(['card_id','month_lag']).agg(agg_func)
agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
agg_history.reset_index(inplace=True)


# In[ ]:


tmp = agg_history.loc[agg_history['month_lag'] == 0]
tmp = tmp[['card_id']].reset_index(drop = True)

for i in range(-13,-1):
    LAG = agg_history.loc[agg_history['month_lag'] == i ]
    lag = agg_history.loc[agg_history['month_lag'] == i+2]
    
    LAG = LAG.merge(lag,how = 'left',on = 'card_id')
    
    LAG['ratio_month_lag_purchase_amount'+'_'+str(i)+'_'+str(i+2)] = LAG['purchase_amount_sum_y']/LAG['purchase_amount_sum_x']
#     LAG['ratio_month_lag_installments'+'_'+str(i)+'_'+str(i+2)] = LAG['installments_sum_y']/LAG['installments_sum_x']
    
    LAG = LAG[['card_id','ratio_month_lag_purchase_amount'+'_'+str(i)+'_'+str(i+2)]]
    
    tmp = tmp.merge(LAG,on = "card_id",how = "left")


# In[ ]:


tmp['average_increase_rate_by_2months'] = tmp.iloc[:,1:].sum(axis = 1)


# In[ ]:


tmp.to_csv("monthly_rate.csv",index=False)

