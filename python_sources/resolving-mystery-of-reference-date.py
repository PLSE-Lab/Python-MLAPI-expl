#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

RAW_DATA = "../input"
hist = pd.read_csv(f'{RAW_DATA}/historical_transactions.csv', low_memory=False, 
                     parse_dates=["purchase_date"])
new =  pd.read_csv(f'{RAW_DATA}/new_merchant_transactions.csv', low_memory=False, 
                     parse_dates=["purchase_date"])

train = pd.read_csv(f'{RAW_DATA}/train.csv', low_memory=False, 
                     parse_dates=["first_active_month"])

test = pd.read_csv(f'{RAW_DATA}/test.csv', low_memory=False, 
                     parse_dates=["first_active_month"])


# ## Purchase dates overlap

# In[ ]:


hist.purchase_date.min(), hist.purchase_date.max(),new.purchase_date.min(), new.purchase_date.max()


# Histoiric transactions start 2017-01-01 00:00:08 and end 2018-02-28 23:59:51   

# ### Checking time duration of the datasets in months

# In[ ]:


from dateutil import rrule
len(list(rrule.rrule(rrule.MONTHLY, dtstart=hist.purchase_date.min(), until=hist.purchase_date.max())))


# In[ ]:


len(list(rrule.rrule(rrule.MONTHLY, dtstart=new.purchase_date.min(), until=new.purchase_date.max())))


# - Histoiric transactions start 2017-01-01 00:00:08 and goes on for 14 months
# - New transactions start on 2017-03-01 03:24:51 and goes on for 14 months
# 
# It looks like there is a lag of 2 months between the datasets.

# ### Duration of the overlap between the datasets 

# In[ ]:


len(list(rrule.rrule(rrule.MONTHLY, dtstart=new.purchase_date.min(), until=hist.purchase_date.max())))


# ## **Transactions per month**

# In[ ]:


hist['month']= [x.strftime("%y-%m") for x in hist.purchase_date]
counts = hist.month.value_counts().sort_index()
counts.plot(kind='bar')


# In[ ]:


new['month']= [x.strftime("%y-%m") for x in new.purchase_date]
counts = new.month.value_counts().sort_index()
counts.plot(kind='bar')


# * Historic transaction count reaches maximun in December of 2017 and then starts to decline for two months.
# * New transactions count spikes in March of 2018, the month when historic tranactions are no longer collected.
# Does it mean that the reference date used in the computation of the month lag is somewhere between December 1st 2017 to January 1st 2018?
# 

# 
# **Let's check the month lag values in the new and historic tranascations**

# In[ ]:


hist.month_lag.min(), hist.month_lag.max() , new.month_lag.min(), new.month_lag.max()


# These numbers contradict the observation. It seems that the spike in sales in december of 2017 is due to seasonal demand.
# The reference date seems to be February 1st 2018. Sales in historic transactions that happened during February of 2018 suppose to have month_lag == 0 
# Tne month_lag for new transactions was not computed prior to  March 1st of 2018, hance the month lag is positive 1 and 2 for the new transactions 
# Spike in sales reflected in the new_merchant transactions in March of 2018 seems to be due to promotion of new merchants that started on the reference date. 
# The spike in sales to new merchants perhaps scared ELO and that prompted this competition, perhaps because that the core merchants in the historic transactions took a serious hit.
# 
# This would be a nice conclusion, but it does not seem to hold given what you are going to see next.

# ### **Number of new_merchant transactoins per day and per month **

# In[ ]:


tmp = new.loc[new.month_lag == 1, 'purchase_date'].value_counts() < 100
new.loc[tmp.values, 'purchase_date'].min(), new.loc[tmp.values, 'purchase_date'].max()


# In[ ]:


pd.DataFrame(new.loc[new.month_lag == 1, 'month'].value_counts()).sort_index().plot(
    title="Number of Transactions with month_lag == 1")


# ###  **Number of card_id counts and  new_merchants transactions counts per month**

# In[ ]:


new.groupby(['month', 'month_lag']).agg({'card_id': pd.Series.nunique }).T


# In[ ]:


cnts = new.groupby([ 'month_lag', 'month']).agg({'card_id': pd.Series.nunique, 'purchase_date' : 'count'})
cnts.columns=['number of cards', 'transaction count']


# In[ ]:


cnts.loc[1,:]


# In[ ]:


cnts.loc[1,:].plot(title = "monthly counts for month_lag=1")


# ## **Card_id monthly salles in hist and new transactions**

# In[ ]:


new_card_ids = new.groupby([ 'month_lag', 'month']).agg({'card_id': list })
hist_cards_monthly_purchase_count = hist.groupby(['card_id', 'month']).agg({'purchase_date': 'count'})


# In[ ]:


hist_cards_monthly_purchase_count.head(15).T


# In[ ]:


prom_cards = list(set(new_card_ids.loc[1].loc['17-03'].values[0]))
hist_card_monthly_purchase_count.loc[prom_cards].T.head()


# In[ ]:


prom_cards = list(set(new_card_ids.loc[1].loc['17-07'].values[0]))
hist_card_monthly_purchase_count.loc[prom_cards].T.head()


# 
# 

# New merchant transactions file contains transactions for cards that were selected for promotions that were suggesting new merchants to existing customers. ELO ran  promotions for new merchants at slow pace since March 1st of 2017 gradually increasing exposure. In March of 2017 number of cards with month_lag == 1 was 213. By January 1st of 2018 there were 18696 cards and by February of 2018  the promotion went out to 173574 card_ids
# 
# On March 1st 2018 the promotion was released to a broader population . Sales to new merchants spiked and historic merchants were on declining trend, but we do not have data in historic transactoins to substantiate that . I guess that this spike led  ELO to a decision that the loyalty score that they have created does not explain true customer loyalty behavior. There was a need for new way of computing loyalty score differently. ELO decided to do this competition. 

# ## Two month difference between the historic and new transactions datasets

# The historic transactions file contain sales leading up to the month when promotion started, but omits data about historic transactions during the time period covered in new_transactions.  This explains the two months lag between historic_treansactions and new_transacitions.
# 
# It seems that two month lag between the datasets might be explained be a two month data collection window prior to promotion. The historic_transactions dataset was used for creating the initial model that led to creation of the first promotion. Once promotion was launched to a (selected) group of customers, the new_transactions dataset was started. The new_transactions dataset (perhaps) contains  data about all card_ids that were exposed to promotion and transactions outside of the core historic group that were a result of promotion were recorded. Hence the new_transaction dataset (perhaps) contains record of all card_ids that completed transactions as a result of promotion. 
# 
# 

# *Please upvote if you find this kernel mildly intersting or entertaining*

# In[ ]:




