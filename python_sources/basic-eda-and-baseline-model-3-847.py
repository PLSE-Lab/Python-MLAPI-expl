#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#df_merchants=pd.read_csv("../input/merchants.csv")
df_historical_transactions=pd.read_csv("../input/historical_transactions.csv")
df_train=pd.read_csv("../input/train.csv")


# In[ ]:


df_test=pd.read_csv("../input/test.csv")


# ## **Exploration and building features**

# ### **First Active Months: Month of first purchase. Lets see the trend of how many first purchases from beginning**

# In[ ]:


df_train.groupby('first_active_month').count()['card_id'].plot(kind='bar',figsize=(40,15))


# This shows number of subscribers increasing. As from wikipedia, Elo was founded in 2010 so this shows trend of subscriber growing from that time. It would be interesting to see how many of old customers are still there performing transactions and whats their loyalty score. We will explore this part later on

# 
# Lets also validate this distribution using test set

# In[ ]:


df_test.groupby('first_active_month').count()['card_id'].plot(kind='bar',figsize=(40,15))


# Distribution looks similar. 

# ### **Lets look at target variable**

# In[ ]:


df_train['target'].describe()


# In[ ]:


df_train.boxplot(column='target', figsize=(20,20))


# From this plot we can see that there are extreme outliers as well if we specifically see minimum value. Mean is therefore effected by this outlier its greater than median. Distribution is skewed from left side.
# . Distribution is tight. Lets also calculate Interquartile(not effected by outliers) to see the range.

# In[ ]:


df_train['target'].describe()['75%'] - df_train['target'].describe()['25%']


# So most lies in this range
# 

# **Lets also relate first active month with target **

# In[ ]:


df_train.head()


# In[ ]:


df_train.sort_values('first_active_month').groupby(['first_active_month']).mean()['target'].plot(kind='bar',figsize=(20,10))


# Why lowest score for customers whose first active month is april 2012?? 

# #### Lets also check how feature(feature-1,feature-2,feature-3) column relates to target

# In[ ]:


fig, ax = plt.pyplot.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
for feature in df_train['feature_1'].unique():
        sns.distplot(df_train[df_train['feature_1']==feature]['target']);


# Distribution looks same for feature-1 column. There are many near 0.
# 
# Model hopefully will be able to relate feature variable and target

# Violen Plot allows us to compare in a more better way. Lets plot for feature 1, feature 2 and feature 3

# In[ ]:


fig, ax = plt.pyplot.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.violinplot(x="feature_1", y="target",  data=df_train, palette="muted",inner="points")


# In[ ]:


fig, ax = plt.pyplot.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.violinplot(x="feature_2", y="target",  data=df_train, palette="muted",inner="points")


# In[ ]:


fig, ax = plt.pyplot.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.violinplot(x="feature_3", y="target",  data=df_train, palette="muted",inner="points")


# These all distribution looks similar. Probability of target being near 0 is higher so Model should be able to find relationship here. We will keep this feature

# ## **Historical Transactions**

# ##### Lets explore and relate historical transaction with target variable

# In[ ]:


df_historical_transactions.head()


# In[ ]:


print("min purchase date", df_historical_transactions['purchase_date'].min())
print("max purchase date", df_historical_transactions['purchase_date'].max())


# We will use this later to calculate dormancy

# In[ ]:


df_historical_transactions['purchase_date'] = pd.to_datetime(df_historical_transactions['purchase_date'])
max_purchase_date=df_historical_transactions['purchase_date'].max()


# There are transactions ranging from 2017 to 2018. Lets now check # of transactions and relate with loyalty of customer

# In[ ]:


g=df_historical_transactions[['card_id']].groupby('card_id')


# In[ ]:


df_transaction_counts=g.size().reset_index(name='num_transactions')


# ### **Include num_transactions as feature**

# In[ ]:


#df_transaction_counts.head()
df_train=pd.merge(df_train,df_transaction_counts, on="card_id",how='left')


# In[ ]:


df_test=pd.merge(df_test,df_transaction_counts, on="card_id",how='left')


# In[ ]:


df_train['num_transactions'].describe()


# ### **Lets bin num transactions and plot to see the relationship with loyalty score**

# In[ ]:


bins=[0,500,1000,1500,2000,2500]


# In[ ]:


df_train['binned_numtransactions']=pd.cut(df_train['num_transactions'],bins)


# In[ ]:


df_train.head()


# In[ ]:


fig, ax = plt.pyplot.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.boxplot(x='binned_numtransactions',y='target',data=df_train)


# It seems that some of lower transaction counts has lower loyalty score as can be seen in above distribution. There are  extreme values from negative side for lower transaction counts. Model will be able to find some relationship here

# ### **Lets now see the trend of purchase date and purchase amount**

# ##### Convert to date and make column YearMonth_Purchase on historical transaction

# In[ ]:


#this takes more memory (not needed now)
#df_historical_transactions['YearMonth_Purchase'] = df_historical_transactions['purchase_date'].map(lambda x: 100*x.year + x.month)


# In[ ]:


#You can plot and check this , there is peak in one particular month, skipping this as it takes memory
#g=df_historical_transactions[['YearMonth_Purchase','purchase_amount']].groupby('YearMonth_Purchase').mean()
#g.plot(kind='bar',figsize=(20,10))


# Interesting. There is peak in one particular Month: April 2017 . We keep this for later part.

# ### **Top 20 Merchants with higher purchase amount**

# In[ ]:


top_merchants_by_purchaseamount=df_historical_transactions[['merchant_id','purchase_amount']].groupby(by='merchant_id').mean().sort_values(by='purchase_amount',ascending=False).head(20)


# In[ ]:


top_merchants_by_purchaseamount.head()


# Skipping this part, you can check top 20 merchants by month year basis

# In[ ]:



#g=df_historical_transactions[df_historical_transactions['merchant_id'].isin(list(top_merchants_by_purchaseamount.index))][['merchant_id','YearMonth_Purchase','purchase_amount']].groupby(['YearMonth_Purchase','merchant_id']).mean()


# In[ ]:


#g.unstack()


# Description based on above output which is commented. 
# 
# I wanted to create line chart on it but there are many NaN so decided not to create it. However there is one interesting thing to note. We have noticed peak in April 2018 (previous graph) , if we see purchases for merchant with id: M_ID_ee49262ab5, this is influencing a lot. We keep this for later part

# ### **Lets add column on training data :favourite merchant, favourite merchant transaction count **

# In[ ]:


g=df_historical_transactions[['card_id','merchant_id']].groupby(['card_id','merchant_id'])


# In[ ]:


merchantid_counts_percard=g.size()


# In[ ]:


merchantid_counts_percard=pd.DataFrame(merchantid_counts_percard)


# In[ ]:


merchantid_counts_percard.head()


# In[ ]:


merchantid_counts_percard.columns=['num_favourite_merchant']


# In[ ]:


merchantid_counts_percard.head()


# In[ ]:


merchantid_counts_percard=merchantid_counts_percard.sort_values(by='num_favourite_merchant',ascending=False)


# In[ ]:


merchantid_counts_percard=merchantid_counts_percard.groupby(level=0).head(1).reset_index()


# In[ ]:


merchantid_counts_percard.columns=['card_id','favourite_merchant','num_transaction_favourite_merchant']


# ### **add Favourite merchant and Favourite merchant count**

# In[ ]:



df_train=pd.merge( df_train,merchantid_counts_percard,on="card_id",how='left')


# In[ ]:


df_test=pd.merge( df_test,merchantid_counts_percard,on="card_id",how='left')


# ### **aggregate Purchase amount and add sum,mean, max, min as feature**

# In[ ]:


df_historical_transactions['purchase_amount'].describe()


# In[ ]:


g=df_historical_transactions[['card_id','purchase_amount']].groupby('card_id')


# In[ ]:


purchaseamount_agg=g.agg(['sum', 'min','max','std','median','mean'])


# In[ ]:


purchaseamount_agg=purchaseamount_agg.reset_index()


# In[ ]:


purchaseamount_agg.head()


# In[ ]:


df_train=pd.merge( df_train,purchaseamount_agg,on="card_id",how='left')


# In[ ]:


df_train.head()


# In[ ]:


df_test=pd.merge( df_test,purchaseamount_agg,on="card_id",how='left')


# ### **Transform first_active_month and get year and month**

# In[ ]:


df_train["first_active_month"]=pd.to_datetime(df_train["first_active_month"])


# In[ ]:


df_test["first_active_month"]=pd.to_datetime(df_test["first_active_month"])


# In[ ]:


df_train["first_active_yr"]=df_train["first_active_month"].dt.year


# In[ ]:


df_test["first_active_yr"]=df_test["first_active_month"].dt.year


# In[ ]:


df_train["first_active_mon"]=df_train["first_active_month"].dt.month


# In[ ]:


df_test["first_active_mon"]=df_test["first_active_month"].dt.month


# In[ ]:


len(df_train['favourite_merchant'].unique())


# ### **Encode Favourite merchant column**

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_train['favourite_merchant'] = le.fit_transform(df_train['favourite_merchant'] )


# In[ ]:


le = preprocessing.LabelEncoder()
df_test['favourite_merchant'] = le.fit_transform(df_test['favourite_merchant'] )


# ### **Last Active Month**

# In[ ]:


last_active_month=df_historical_transactions.loc[df_historical_transactions.groupby('card_id').purchase_date.idxmax(),:][['card_id','purchase_date','purchase_amount']]


# In[ ]:


last_active_month.columns=['card_id','last_active_purchase_date','last_active_purchase_amount']


# In[ ]:


last_active_month.head()


# In[ ]:


df_train=pd.merge(df_train,last_active_month, on="card_id",how='left')


# In[ ]:


df_test=pd.merge(df_test,last_active_month, on="card_id",how='left')


# In[ ]:


df_train['last_active_purchase_year']=df_train['last_active_purchase_date'].dt.year


# In[ ]:


df_train['last_active_purchase_month']=df_train['last_active_purchase_date'].dt.month


# In[ ]:


df_train['last_active_purchase_day']=df_train['last_active_purchase_date'].dt.day


# In[ ]:


df_test['last_active_purchase_year']=df_test['last_active_purchase_date'].dt.year


# In[ ]:


df_test['last_active_purchase_month']=df_test['last_active_purchase_date'].dt.month


# In[ ]:


df_test['last_active_purchase_day']=df_test['last_active_purchase_date'].dt.day


# ### **Add Dormancy**

# In[ ]:


max_purchase_date


# In[ ]:



df_train['dormancy']=[(max_purchase_date-x).days for x in df_train['last_active_purchase_date']]


# In[ ]:


df_test['dormancy']=[(max_purchase_date-x).days for x in df_test['last_active_purchase_date']]


# In[ ]:


df_train.head()


# In[ ]:


df_test.columns


# ## **Training Model**

# In[ ]:




from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# In[ ]:


df_train.columns=['first_active_month','card_id','feature_1','feature_2','feature_3','target','num_transactions',
                  'binned_numtransactions','favourite_merchant','num_transaction_favourite_merchant',
                  'sum_purchase_amount','min_purchase_amount','max_purchase_amount',
                  'std_purchase_amount','median_purchase_amount','mean_purchase_amount',
                  'first_active_yr','first_active_mon','last_active_purchase_date',
       'last_active_purchase_amount', 'last_active_purchase_year','last_active_purchase_month', 
                  'last_active_purchase_day', 'dormancy']


# In[ ]:


df_test.columns=['first_active_month','card_id','feature_1','feature_2','feature_3',
                 'num_transactions','favourite_merchant','num_transaction_favourite_merchant',
                 'sum_purchase_amount','min_purchase_amount','max_purchase_amount',
                 'std_purchase_amount','median_purchase_amount','mean_purchase_amount','first_active_yr','first_active_mon','last_active_purchase_date',
            'last_active_purchase_amount', 'last_active_purchase_year',
           'last_active_purchase_month', 'last_active_purchase_day', 'dormancy']


# In[ ]:


df_test.head()


# In[ ]:


final_cols=['feature_1','feature_2','feature_3','num_transactions','favourite_merchant','num_transaction_favourite_merchant','sum_purchase_amount',
            'min_purchase_amount','max_purchase_amount','std_purchase_amount','median_purchase_amount','mean_purchase_amount',
            'first_active_yr','first_active_mon','last_active_purchase_amount', 'last_active_purchase_year',
           'last_active_purchase_month', 'last_active_purchase_day', 'dormancy']
target_col=['target']


# In[ ]:



lgb_params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }


# ### **Using 7 K Fold cross validation**

# In[ ]:


Folds = KFold(n_splits=7, shuffle=True, random_state=1989)


# In[ ]:



pred_train = np.zeros(len(df_train))
pred_test = np.zeros(len(df_test))

features_lgb = list(df_train.columns)
feature_importance = pd.DataFrame()


# ### **Prepare Data (X and Y for training)**

# In[ ]:


train_X=df_train[final_cols]


# In[ ]:


train_y=df_train[target_col]


# In[ ]:


test_X=df_test[final_cols]


# In[ ]:


for fold_, (train_idx, val_idx) in enumerate(Folds.split(df_train)):
    train_data = lgb.Dataset(train_X.iloc[train_idx], label=train_y.iloc[train_idx])
    val_data = lgb.Dataset(train_X.iloc[val_idx], label=train_y.iloc[val_idx])

    num_round = 10000
    model = lgb.train(lgb_params, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
    pred_train[val_idx] = model.predict(train_X.iloc[val_idx], num_iteration=model.best_iteration)

    pred_test += model.predict(test_X, num_iteration=model.best_iteration) / Folds.n_splits


# In[ ]:



print(np.sqrt(mean_squared_error(pred_train, df_train[target_col])))


# In[ ]:


submit_df = pd.read_csv('../input/sample_submission.csv')
submit_df["target"] = pred_test
submit_df.to_csv("submission_baseline_lgb.csv", index=False)


# In[ ]:


submit_df


# In[ ]:


fig, ax = plt.pyplot.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.pyplot.title("LGB- Feature Importance", fontsize=15)

