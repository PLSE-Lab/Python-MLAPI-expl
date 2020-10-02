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
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.O
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data=pd.read_csv(r'../input/train.csv')
test_data=pd.read_csv(r'../input/test.csv')


# In[ ]:


get_ipython().system('../input/Data_Dictionary.xlsx')


# In[ ]:


dict1=pd.read_excel(r'../input/Data_Dictionary.xlsx')
dict1


# In[ ]:


train_data.head(20)


# In[ ]:


train_data.tail()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe(include='all')


# In[ ]:


train_data.isnull().sum()


# In[ ]:


plt.figure(figsize=(8,6),dpi=80)
sns.violinplot(x=train_data['feature_1'],y=train_data['target'],data=train_data,)
plt.xlabel('Target',size=18,color='r')
plt.ylabel('Feature_1',size=18,color='r')
plt.title('Feature_1 Vs Target',size=20,color='blue')
plt.legend();


# In[ ]:


plt.figure(figsize=(8,6),dpi=80)
sns.violinplot(x=train_data['feature_2'],y=train_data['target'],data=train_data,)
plt.xlabel('Target',size=18,color='r')
plt.ylabel('Feature_2',size=18,color='r')
plt.title('Feature_2 Vs Target',size=20,color='blue')
plt.legend();


# In[ ]:


plt.figure(figsize=(8,6),dpi=80)
sns.violinplot(x=train_data['feature_3'],y=train_data['target'],data=train_data)
plt.xlabel('Target',size=18,color='r')
plt.ylabel('Feature_3',size=18,color='r')
plt.title('Feature_3 Vs Target',size=20,color='blue')
plt.legend();


# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(train_data['target'],bins=50,hist=True,color='#F71212',kde=False)


# As we can see that some targets are below** (-)30** lets check how many are there 

# In[ ]:


print('there are {0} sample in target below -30'.format(train_data.loc[train_data.target<-30].shape[0]))


# In[ ]:


feature1=train_data['feature_1'].value_counts().sort_index(ascending=False)
sns.barplot(x=feature1.index,y=feature1.values,ci=100,color='#FFFF00')
plt.title('Feature_1',color='red',size=18);


# In[ ]:


feature2=train_data['feature_2'].value_counts().sort_index(ascending=False)
sns.barplot(x=feature2.index,y=feature2.values,ci=100,color='#00FF7F')
plt.title('Feature_2',color='red',size=18);


# In[ ]:


feature3=train_data['feature_3'].value_counts().sort_index(ascending=False)
sns.barplot(x=feature3.index,y=feature3.values,ci=100,color='#556B2F')
plt.title('Feature_3',color='red',size=18);


# As we can see that different feature have different distribution of data according to target 
# 1. feature_1 value varied from 1 to 5 in target 
# 2. feature_1 value varied from 1 to 3 in target 
# 3. feature_1 value varied from 0 to 1 in target 

# In[ ]:


train_active_months=train_data['first_active_month'].value_counts().sort_index(ascending=False)
test_active_months=test_data['first_active_month'].value_counts().sort_index(ascending=False)


# In[ ]:


data = [go.Scatter(x=train_active_months.index, y=train_active_months.values, name='train',opacity=1), 
        go.Scatter(x=test_active_months.index, y=test_active_months.values, name='test',opacity=1)]
layout = go.Layout(dict(title = "Counts of first active month",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Count'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))


# As we can see that both train and test active_month data follow the same trends and thats great !!

# In[ ]:


test_data.head()


# In[ ]:


test_data.describe()


# In[ ]:


test_data.info()


# In[ ]:


test_data.isnull().sum()


# Ahh!!! one missing value in first_active_month in test data 

# In[ ]:


historical_data=pd.read_csv(r'../input/historical_transactions.csv')


# In[ ]:


historical_data.head()


#  *** The field descriptions are as follow:**
# 1. card_id - Card identifier
# 2. month_lag - month lag to reference date
# 1. category_1 - anonymized category
# 1. category_2 - anonymized category
# 1. category_3 - anonymized category
# 3. purchase_date - Purchase date
# 4. authorized_flag - 'Y' if approved, 'N' if denied
# 1. installments - number of installments of purchase
# 1. merchant_category_id - Merchant category identifier (anonymized )
# 1. merchant_id - Merchant identifier (anonymized)
# 1. purchase_amount - Normalized purchase amount
# 1. city_id - City identifier (anonymized )
# 1. state_id - State identifier (anonymized )
# 1.  subsector_id - Merchant category group identifier (anonymized )
# 

# In[ ]:


historical_data.tail()


# In[ ]:


historical_data.info()


# In[ ]:


historical_data.isnull().sum()


# In[ ]:


historical_data['installments'].value_counts()


# In[ ]:


plt.figure(figsize=(8,6),dpi=100)
install=historical_data['installments'].value_counts().sort_index(ascending=False)
sns.barplot(x=install.index,y=install.values,ci=100,color='#900C3F')
plt.title('installment detail',color='red',size=18);


# In[ ]:


trai=historical_data.groupby(['installments'])['authorized_flag'].value_counts()
trai.head()


# In[ ]:


new_merchant_data=pd.read_csv(r'../input/new_merchant_transactions.csv')


# In[ ]:


new_merchant_data.head()


# In[ ]:


new_merchant_data.info()


# In[ ]:


new_merchant_data.isna().sum()


# category_2 follow trends of **->** **1.0 **
# category_3 follow ->A,B or C

# In[ ]:


new_merchant_data['category_2'].fillna(value=1.0,inplace=True)
new_merchant_data['category_3'].fillna('A',inplace=True)
new_merchant_data['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


# In[ ]:


historical_data['category_2'].fillna(value=1.0,inplace=True)
historical_data['category_3'].fillna('A',inplace=True)
historical_data['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


# In[ ]:


new_merchant_data['authorized_flag']=new_merchant_data['authorized_flag'].apply(lambda x:1 if x=='Y'else 0)


# In[ ]:


new_merchant_data['authorized_flag'].value_counts().plot(kind='bar',title='authorize_flag value',color='red')


# All Transcation looks authorized 

# In[ ]:


card_total=new_merchant_data.groupby(['card_id'])['purchase_amount'].mean().sort_values()
card_total.head()


# In[ ]:


def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


# In[ ]:


import datetime
for df in [historical_data,new_merchant_data]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']


# In[ ]:


aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    historical_data[col+'_mean'] = historical_data.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']    

new_columns = get_new_columns('hist',aggs)
historical_data_group = historical_data.groupby('card_id').agg(aggs)
historical_data_group.columns = new_columns
historical_data_group.reset_index(drop=False,inplace=True)
historical_data_group['hist_purchase_date_diff'] = (historical_data_group['hist_purchase_date_max'] -historical_data_group['hist_purchase_date_min']).dt.days
historical_data_group['hist_purchase_date_average'] = historical_data_group['hist_purchase_date_diff']/historical_data_group['hist_card_id_size']
historical_data_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - historical_data_group['hist_purchase_date_max']).dt.days


# In[ ]:


#merge with train, test
train_data = train_data.merge(historical_data_group,on='card_id',how='left')
test_data = test_data.merge(historical_data_group,on='card_id',how='left')

#cleanup memory
del historical_data_group;


# In[ ]:


aggre = {}
for col in ['subsector_id','merchant_id','merchant_category_id','state_id', 'city_id']:
    aggre[col] = ['nunique']
for col in ['month', 'hour', 'weekofyear', 'dayofweek']:
    aggre[col] = ['nunique', 'mean', 'min', 'max']    
aggre['purchase_amount'] = ['sum','max','min','mean','var']
aggre['installments'] = ['sum','max','min','mean','var']
aggre['purchase_date'] = ['max','min']
aggre['month_lag'] = ['max','min','mean','var']
aggre['month_diff'] = ['mean']
aggre['weekend'] = ['sum', 'mean']
aggre['category_1'] = ['sum', 'mean']
aggre['card_id'] = ['size']


# In[ ]:


for col in ['category_2','category_3']:
    new_merchant_data[col+'_mean'] = new_merchant_data.groupby([col])['purchase_amount'].transform('mean')
    new_merchant_data[col+'_min'] = new_merchant_data.groupby([col])['purchase_amount'].transform('min')
    new_merchant_data[col+'_max'] = new_merchant_data.groupby([col])['purchase_amount'].transform('max')
    new_merchant_data[col+'_sum'] = new_merchant_data.groupby([col])['purchase_amount'].transform('sum')
    new_merchant_data[col+'_std'] = new_merchant_data.groupby([col])['purchase_amount'].transform('std')
    aggre[col+'_mean'] = ['mean']


# In[ ]:


new_columns = get_new_columns('new_hist',aggre)
new_merchant_data_group = new_merchant_data.groupby('card_id').agg(aggre)
new_merchant_data_group.columns = new_columns
new_merchant_data_group.reset_index(drop=False,inplace=True)
new_merchant_data_group['new_hist_purchase_date_diff'] = (new_merchant_data_group['new_hist_purchase_date_max'] - new_merchant_data_group['new_hist_purchase_date_min']).dt.days
new_merchant_data_group['new_hist_purchase_date_average'] = new_merchant_data_group['new_hist_purchase_date_diff']/new_merchant_data_group['new_hist_card_id_size']
new_merchant_data_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - new_merchant_data_group['new_hist_purchase_date_max']).dt.days
new_merchant_data_group['new_hist_purchase_date_uptomin'] = (datetime.datetime.today() - new_merchant_data_group['new_hist_purchase_date_min']).dt.days
#merge with train, test


# In[ ]:


train_data = train_data.merge(new_merchant_data_group,on='card_id',how='left')
test_data = test_data.merge(new_merchant_data_group,on='card_id',how='left')


# In[ ]:


del new_merchant_data_group;
del historical_data;
del new_merchant_data;


# In[ ]:


train_data.head()


# In[ ]:


train_data['outliers'] = 0
train_data.loc[train_data['target'] < -30, 'outliers'] = 1
train_data['outliers'].value_counts()


# In[ ]:


for df in [train_data,test_data]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['quarter'] = df['first_active_month'].dt.quarter
    df['is_month_start'] = df['first_active_month'].dt.is_month_start
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_last_buy'] = (df['new_hist_purchase_date_max'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',                     'new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9


# In[ ]:


df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']
df['purchase_amount_mean'] = df['new_hist_purchase_amount_mean']+df['hist_purchase_amount_mean']
df['purchase_amount_max'] = df['new_hist_purchase_amount_max']+df['hist_purchase_amount_max']
for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',                     'new_hist_purchase_date_min']:
    df[f] = df[f].astype(np.int64) * 1e-9
df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

for f in ['feature_1','feature_2','feature_3']:
    order_label = train_data.groupby([f])['outliers'].mean()
    train_data[f] = train_data[f].map(order_label)
    test_data[f] = test_data[f].map(order_label)


# In[ ]:


train_data_columns = [c for c in train_data.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = train_data['target']
del train_data['target']


# In[ ]:


train_data.head()


# In[ ]:


param = {'num_leaves': 51,
         'min_data_in_leaf': 35, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.008,
         "boosting": "gbdt",
         "feature_fraction": 0.85,
         "bagging_freq": 1,
         "bagging_fraction": 0.82,
         "bagging_seed": 42,
         "metric": 'rmse',
         "lambda_l1": 0.11,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 2019}
#prepare fit model with cross-validation
folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=2019)
oof = np.zeros(len(train_data))
predictions = np.zeros(len(test_data))
feature_importance_df = pd.DataFrame()
#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data,train_data['outliers'].values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    trn_data = lgb.Dataset(train_data.iloc[trn_idx][train_data_columns], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_data.iloc[val_idx][train_data_columns], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
    oof[val_idx] = clf.predict(train_data.iloc[val_idx][train_data_columns], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_data_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions += clf.predict(test_data[train_data_columns], num_iteration=clf.best_iteration) / folds.n_splits
    
strRMSE = "".format(np.sqrt(mean_squared_error(oof, target)))
print(strRMSE)


# In[ ]:


submission_data = pd.DataFrame({"card_id":test_data["card_id"].values})
submission_data["target"] = predictions
submission_data.to_csv("baseline_lgb1.csv", index=False)


# In[ ]:





# ****Many more to come stay tuned!!!**
# 
# **DON'T FORGET TO UPVOTE :)****

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




