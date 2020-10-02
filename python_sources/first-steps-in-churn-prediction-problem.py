#!/usr/bin/env python
# coding: utf-8

# 
# #Introduction#
# 
# In this kernel are presented data exploration and preparation for [WSDM - KKBox's Churn Prediction Challenge](http://https://www.kaggle.com/c/kkbox-churn-prediction-challenge/) the goal of which is to build a predictive model to classify users between 2 classes: churn (1) and non-churn (0) based their activity and payment history along all their lifetime on service.
# 
# ![Model](https://i.imgur.com/ep0WSb0.png)
# 

# As input data for builing model we have:
# * members_v2.csv - dataset with users of KKbox with registration date from April 2014 till April 2017 (6.7M users)
# * train.csv  - users marked by churn attribute in March 2017
# * transactions.csv - payment information about users from Jan 2015 till Feb 2017
# * user_logs.csv - user behavior (% and number of listened songs)
# 
# All datasets will be merged on 'msno' - user identification and possible that not all user from members.csv will be presented in other sets and vice versa.
# 
# I will use *Python 3.6 with Pandas.*
# 
# #Quick Data Exploration#
# 
# ##train.csv##
# 
# First of all let make superficial look on data sets and begin from *train.csv.*
# Churn rate in Feb 2017 was 6.4% against 93.6% regular users.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from datetime import datetime


# In[ ]:


train=pd.read_csv('../input/train.csv')
gr=train.groupby('is_churn').count()
gr.plot.pie(subplots=True,autopct='%.2f',figsize=(5, 5))


# Check dataset for data missing and outlier and found that data is clear.

# In[ ]:


train.isnull().sum()


# ##Members.csv##
# 

# In[ ]:


members=pd.read_csv('../input/members_v2.csv')
members.head()


# Check *members.csv* for missing data:

# In[ ]:


members.isnull().sum()


# Looks like *gender* field has a large number of missing data. I prefer to exclude that field from input data for the predictive model.

# In[ ]:


members=members.fillna("NA")
members.groupby('gender').count().plot.bar(y='msno',figsize=(16, 5))


# Brief look to other column shows that '*bd'* (birthday field) include nubmer of outliers and more then half of not relevant data (age<0).
# 'bd' filed will be excludet from predictive model as a 'gender'. 
# 

# In[ ]:


members.hist(figsize=(16, 10))


# In[ ]:


bin=list(range(-3200,1980, 100))
group=members.groupby(pd.cut(members.bd, bin)).count()
group.plot.bar(y='msno',figsize=(16, 5))


# After data  munging *merge.csv* will look like:

# In[ ]:


columns = ['gender','bd']
members=members.drop(columns, axis=1)


# In[ ]:


members['registration_init_time']=pd.to_datetime(members['registration_init_time'],format='%Y%m%d')


# In[ ]:


members.head()


# ##Transactions.csv##
# 
# Will check *transactions.csv* for missing data, outliers and not-meaningful attributes.

# In[ ]:


transactions = pd.read_csv('../input/transactions.csv')
transactions.head()


# Checking for missing data shown that all field filled.

# In[ ]:


transactions.isnull().sum()


# Converting date filds into timestamp

# In[ ]:


transactions['transaction_date']=pd.to_datetime(transactions['transaction_date'],format='%Y%m%d') 
transactions['membership_expire_date']=pd.to_datetime(transactions['membership_expire_date'],format='%Y%m%d')


# Numerical data looks clean:  without outliers.

# In[ ]:


transactions.describe().transpose()


# As we can see from plots most of the transactions are auto-renewal subscriptions, a number of canceled transaction is not big.

# In[ ]:


transactions.hist(column=['is_cancel', 'is_auto_renew'],figsize=(16, 5))


# Most of users use payment method encoded as "40" and payment plan days "30 days" is absolute leader.

# In[ ]:


transactions.hist(column=['payment_method_id','payment_plan_days'],figsize=(16, 5),bins=50)


# Plots for *"plan list price"* and *"actual amount paid"* are identical with median=149.

# In[ ]:


transactions.hist(column=['actual_amount_paid','plan_list_price'],figsize=(16, 5),bins=50)


# In[ ]:


transactions["actual_amount_paid"].median()


# In[ ]:


transactions["plan_list_price"].median()


# ##User_logs.csv##
# 
# User_logs.csv size is 6.65 GB in archive format. For this trainnig kernal I will read a part of it - 30M rows.

# In[ ]:


logs = pd.read_csv('../input/user_logs.csv', nrows=30000000)


# In[ ]:


logs.head()


# Checking data for missing valuas  and converting  date format into timestamp.

# In[ ]:


logs.isnull().sum()


# In[ ]:


logs['date']=pd.to_datetime(logs['date'],format='%Y%m%d') 


# According to plot below most songs are listened to the end every day.

# In[ ]:


gr=logs.groupby(pd.Grouper(key='date', freq='D')).mean()


# In[ ]:


gr.plot.line(y=['num_25','num_50','num_75','num_985','num_100'],figsize=(16, 5))


# Average number of unique songs played per day by users.

# In[ ]:


gr.plot.line(y=['num_unq'],figsize=(16, 5))


# #Hypothesis generation#
# 
# First useful information dataset will be merged from *transaction.csv, train.csv *and* members.csv.* with 11.7M rows with information about transactions of ~700K users

# In[ ]:


train_total=pd.merge(transactions,pd.merge(members,train,on='msno',how='inner'),on='msno',how='inner')
train_total.head()


# In[ ]:


train_total.info()


# In[ ]:


gr=train_total.groupby(['transaction_date','is_churn']).count()


# Below you can see on the plot number of transactions per user (chart once again shows the popularity of 30-day auto-renewal subscriptions)

# In[ ]:


gr2=gr.unstack()
gr2.plot.line(y='msno',figsize=(16, 5))


# If look on users who churn on March 2017 we can see some transactions in February 2017...

# In[ ]:


gr2.plot.line(y='msno',figsize=(20, 8),ylim=(0,600),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))


# Transactions in February from churned user was "canceled transactions" only.

# In[ ]:


df1=train_total[train_total['is_cancel'] ==1]


# In[ ]:


gr3=df1.groupby(['transaction_date','is_churn']).sum()


# In[ ]:


gr4=gr3.unstack()
gr4.plot.line(y='is_cancel',figsize=(16, 5),ylim=(0,60),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))


# Second useful dataset will be merged from *user_logs.csv, train.csv *and* members.csv.* ~14.3M rows.

# In[ ]:


logs_total=pd.merge(logs,pd.merge(members,train,on='msno',how='inner'),on='msno',how='inner')
logs_total.head()


# In[ ]:


logs_total.info()


# Number of unique songs played 100% by users depending on churn attribute.

# In[ ]:


group1=logs_total.groupby(['date','is_churn']).sum()


# In[ ]:


group2=group1.unstack()
group2.plot.line(y='num_100',figsize=(16, 5))


# In[ ]:


group2.plot.line(y=['num_100'],figsize=(16, 5),ylim=(0,15000),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))


# Unfortunately, I lost data about churned user behavior when decided to limit file importation. And now I see lack of data for data-driven hypothesis generation.

# In[ ]:


group2.plot.line(y=['num_50'],figsize=(16, 5),ylim=(0,2000),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))


# Only, from logic and experience, I can guess that *user_logs.csv* will be more useful for predict user involment or satisfaction in service and *transactions.csv* can show financial pattern for usage and possible user revival after some churn period (seasonality etc.)

# > I am new to Python and real big data analysis, and I will be thankful for advice, remarks or constructive criticism.
