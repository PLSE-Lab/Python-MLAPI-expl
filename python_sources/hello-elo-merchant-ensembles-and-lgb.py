#!/usr/bin/env python
# coding: utf-8

# In this Notebook try to explore the data given for **Elo Merchant Category Recommendation** , let us compile about the compitation 

# **Elo**
# 
# One of the largest payment brands in Brazil, has built partnerships with merchants in order to offer promotions or discounts to cardholders.

# **Objective:**
# 
# Build a model which will improve RMSE error in the predictions
# 
# 

# **What am I predicting?**
# 
# You are predicting a loyalty score for each card_id represented in test.csv and sample_submission.csv.

# **Data Exploration**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


hist_trans=pd.read_csv('../input/historical_transactions.csv')
merchants=pd.read_csv('../input/merchants.csv')
new_merchant_transactions=pd.read_csv('../input/new_merchant_transactions.csv')


# In[ ]:


train_df=pd.read_csv('../input/train.csv',parse_dates=["first_active_month"])
test_df=pd.read_csv('../input/test.csv',parse_dates=['first_active_month'])

print('Number of rows and columns : ',train_df.shape)
print('Number of rows and columns : ',test_df.shape)


# In[ ]:


train_df.head()


# Lets understand how target column for given dataset

# In[ ]:


target_col = "target"

plt.scatter(range(train_df.shape[0]),np.sort(train_df.target.values))

plt.xlabel("Index",fontsize=12)
plt.ylabel("LoyalityScore",fontsize=12)

plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
plt.hist(train_df.target.values,bins=50,color='red')

plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()


# In[ ]:


(train_df.target.values < -30).sum()


# We can notice some extreme outlier compare to actual Loyality score, its close to 1% of acutal data . 

# Lets look at how 

# In[ ]:


cnt_srs = train_df['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = test_df['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()




# In[ ]:


sns.violinplot(x='feature_1',y=train_df.target.values,data=train_df)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()

sns.violinplot(x='feature_2',y=train_df.target.values,data=train_df)
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()

sns.violinplot(x='feature_3',y=train_df.target.values,data=train_df)
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()


# Not much difference in there distribution of traing and testing data.

# **Finding missing values from Train and Test**

# In[ ]:


#Train Missing Values
total=train_df.isnull().sum().sort_values(ascending = False)
percentage=(train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)

missing_data=pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
missing_data.head(10)


# In[ ]:


#Test missing values
total=test_df.isnull().sum().sort_values(ascending = False)
percentage=(test_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)

missing_data=pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
missing_data.head(10)


# In[ ]:


import datetime


# In[ ]:


import datetime

for df in [train_df,test_df]:
    df['first_active_month']=pd.to_datetime(df['first_active_month'])
    df['year']=df['first_active_month'].dt.year
    df['month']=df['first_active_month'].dt.month
    
    df['elapsed_time']=(datetime.date(2018,2,1)-df['first_active_month'].dt.date).dt.days

target=train_df[target_col]
del train_df['target']


# In[ ]:


train_df.head()


# **Simple Exploration: Historical Transactions**

# In[ ]:


hist_trans.head()


# In[ ]:


print(hist_trans.shape)


# * card_id : Card identifier
# * month_lag : month lag to reference date
# * purchase_date : Purchase date
# * authorized_flag : Y' if approved, 'N' if denied
# * category_3 : anonymized category
# * installments : number of installments of purchase
# * category_1 : anonymized category
# * merchant_category_id : Merchant category identifier (anonymized )
# * subsector_id : Merchant category group identifier (anonymized )
# * merchant_id : Merchant identifier (anonymized)
# * purchase_amount : Normalized purchase amount
# * city_id : City identifier (anonymized )
# * state_id : State identifier (anonymized )
# * category_2 : anonymized category

# In[ ]:


hist_trans['authorized_flag']=hist_trans['authorized_flag'].map({'Y':1,'N':0})


# In[ ]:


hist_trans.loc[:,'purchase_date']=pd.DatetimeIndex(hist_trans['purchase_date']).astype(np.int64)*1e-9

# aggregate function
agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
agg_histroy=hist_trans.groupby(['card_id']).agg(agg_func)
agg_histroy.columns=['hist_' + '-'.join(col).strip() for col in agg_histroy.columns.values]
agg_histroy.reset_index(inplace=True)    


# In[ ]:


df=agg_histroy.groupby(['card_id']).size().reset_index(name='hist_transactions_count')
agg_histroy=pd.merge(df,agg_histroy,on='card_id',how='left')

agg_histroy.head(10)


# In[ ]:


train=pd.merge(train_df,agg_histroy,on='card_id',how='left')
test=pd.merge(train_df,agg_histroy,on='card_id',how='left')


# In[ ]:


train.head(10)


# **Simple Exploration: New_merchants**

# In[ ]:


new_merchant_transactions.head(5)


# In[ ]:


new_merchant_transactions.shape


# * card_id : Card identifier
# * month_lag : month lag to reference date
# * purchase_date : Purchase date
# * authorized_flag : Y' if approved, 'N' if denied
# * category_3 : anonymized category
# * installments : number of installments of purchase
# * category_1 : anonymized category
# * merchant_category_id : Merchant category identifier (anonymized )
# * subsector_id : Merchant category group identifier (anonymized )
# * merchant_id : Merchant identifier (anonymized)
# * purchase_amount : Normalized purchase amount
# * city_id : City identifier (anonymized )
# * state_id : State identifier (anonymized )
# * category_2 : anonymized category

# In[ ]:


new_merchant_transactions['authorized_flag']=new_merchant_transactions['authorized_flag'].map({'Y':1,'N':0})


# In[ ]:


# Finding missing values

total=new_merchant_transactions.isnull().sum().sort_values(ascending=False)
percentage=(new_merchant_transactions.isnull().sum()/new_merchant_transactions.isnull().count() * 100).sort_values(ascending=False)

missing_data=pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

missing_data


# In[ ]:


# Making aggregations on New Merchant transactions

# new_merchant_transactions.loc[:,'purchase_date']=pd.DatetimeIndex(new_merchant_transactions['purchase_date']).astype(np.int64)*1e-9

# aggregate function
new_agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max']
        }
new_agg_histroy=new_merchant_transactions.groupby(['card_id']).agg(new_agg_func)
# new_agg_histroy.head(10)
new_agg_histroy.columns=['new_' + '-'.join(col1).strip() for col1 in new_agg_histroy.columns.values]
new_agg_histroy.reset_index(inplace=True)
new_agg_histroy.head(10)
new_df=(new_agg_histroy.groupby(['card_id']).size().reset_index(name='new_transactions_count'))

new_agg_histroy=pd.merge(new_df,new_agg_histroy,on='card_id',how='left')


# In[ ]:


new_agg_histroy.head(10)


# In[ ]:


train=pd.merge(train,new_agg_histroy,on='card_id',how='left')
test=pd.merge(test,new_agg_histroy,on='card_id',how='left')


# In[ ]:


train.shape,test.shape


# In[ ]:


use_col=[col for col in train.columns if col not in ['card_id', 'first_active_month']]


# We have all featured ready , lets go head and try with base model

# **Base Model**
# 
# kernel : https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you

# In[ ]:


use_col=[col for col in train.columns if col not in ['card_id', 'first_active_month']]

# using below train & test data for our prediction

train=train[use_col]
test=train[use_col]

features=list(train.columns)
categorical_features=[featr for featr in features if 'feature_' in featr]

for category_col in categorical_features:
    print(category_col,'********',train[category_col].value_counts().shape[0],'categories')
    


# Label Encoding for categorical features

# In[ ]:


from sklearn.preprocessing import LabelEncoder
for cat in categorical_features:
    print(cat)
    lbe=LabelEncoder()
    lbe.fit(list(train[cat].values.astype('str')) + list(test[cat].values.astype('str')))
    train[cat]=lbe.transform(list(train[cat].values.astype('str')))
    test[cat]=lbe.transform(list(test[cat].values.astype('str')))


# In[ ]:


train.shape


# In[ ]:


df_all=pd.concat([train,test])
df_all=pd.get_dummies(df_all,columns=categorical_features)
# df_all.head(20)



# In[ ]:


len_train=train.shape[0]
print(len_train)
train=df_all[:len_train]
print(train.shape)
test=df_all[len_train:]
print(test.shape)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


lgb_params = {"objective" : "regression", "metric" : "rmse", 
               "max_depth": 7, "min_child_samples": 20, 
               "reg_alpha": 1, "reg_lambda": 1,
               "num_leaves" : 64, "learning_rate" : 0.005, 
               "subsample" : 0.8, "colsample_bytree" : 0.8, 
               "verbosity": -1}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    

print(np.sqrt(mean_squared_error(oof_lgb, target)))


# In[ ]:


cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = pd.Series(predictions_lgb)
sub_df.to_csv("submission_lgb.csv", index=False)


# **More to Come !!!!**

# 

# 

# 
