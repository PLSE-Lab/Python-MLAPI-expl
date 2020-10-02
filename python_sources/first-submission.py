#!/usr/bin/env python
# coding: utf-8

# Loading all libraries and reading files. We will also parse dates while reading through files.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from plotnine import *
import matplotlib.pyplot as plt
import plotnine
import lightgbm as lgb
from mizani.breaks import date_breaks

print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv', parse_dates =['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates =['first_active_month'])


# In[ ]:


hist_trans = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])
new_trans = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=['purchase_date'])
merchants = pd.read_csv('../input/merchants.csv')


# Let's have a look at shape

# In[ ]:


dfs = [train, test, hist_trans, new_trans, merchants]
for d in dfs:
    print(d.shape)


# Also I would like to explore train and test set with describe command. Describe is my favorite command and it gives really nice glimpse of data. 

# In[ ]:


train.describe(include='all')


# In[ ]:


test.describe(include='all')


# It looks like there is one null value for first activation column for test dataset. Let's fill it with most common date

# In[ ]:


test['first_active_month'] = pd.to_datetime(test['first_active_month'].fillna('2017-09-01 00:00:00'))


# Let's also look at target value distribution in test dataset

# In[ ]:


plotnine.options.figure_size = (14, 6)
ggplot(train, aes(x='target')) +geom_histogram(bins=30, fill='blue', color='black') +scale_x_continuous(breaks=range(-40, 20, 10))


# Most of the values belong to range between -10 and 10.
# 
# It is also good to know whether age of card dependancy on target. For this we need to create new featuer by substracting current date from activation date. We will also explore first active month feature with simple bar chart.

# In[ ]:


train['dayDiffActiveMonth'] = (train['first_active_month'] - pd.to_datetime(datetime.datetime.now().date())).dt.days
test['dayDiffActiveMonth'] = (test['first_active_month'] - pd.to_datetime(datetime.datetime.now().date())).dt.days


# In[ ]:


plotnine.options.figure_size = (14, 6)
ggplot(train, aes(x='first_active_month')) +geom_bar(fill='red', alpha=0.5, color='black') +scale_x_datetime(breaks=date_breaks('1 month')) +theme(axis_text_x=element_text(rotation=90, hjust=1))


# Now let's look at pearson co-relation between variables.

# In[ ]:


train.corr()


# This shows our manufactured feature - "dayDiffActiveMonth" is having highest co-relation with target variable. However there are no strong co-relations. and we need to manufacture other features to make our model strong. So let's explore how we can use historical and new merchant transactions to create new features.

# In[ ]:


hist_trans.describe(include='all')


# In[ ]:


new_trans.describe(include='all')


# Both historical and new merchant data got similar structure. In first phase we will just count number of transactions
# and average of amount. I just want to include numeric features into consideration for now for simplicity. I will add
# categorical feature's details in later part of kernel(if needed)
# 
# Following features seem numeric :  installments, month_lag, purchase_amount 
# 
# I would also like to convert to flags into number so we can consider mean of those binary features.

# In[ ]:


hist_trans.head()


# In[ ]:


hist_trans['authorized_flag'] = hist_trans['authorized_flag'].map({"Y" : 0, "N" : 1})
new_trans['authorized_flag'] = new_trans['authorized_flag'].map({"Y" : 0, "N" : 1})

hist_trans['category_1'] = hist_trans['category_1'].map({"Y" : 0, "N" : 1})
new_trans['category_1'] = new_trans['category_1'].map({"Y" : 0, "N" : 1})


# In[ ]:


for c in ['authorized_flag', 'category_1','installments','month_lag','purchase_amount']:
    hist_transt_amt_summ = hist_trans.groupby(['card_id']).agg({c : ['count', 'sum', 'mean', 'max', 'std']}).reset_index()
    hist_transt_amt_summ.columns = ['card_id', 'hist_'+ c +'_cnt', 'hist_'+ c +'_sum', 
                                    'hist_'+ c +'_mean',  'hist_'+ c +'_max', 'hist_'+ c +'_std']
    train = train.merge(hist_transt_amt_summ, how='left', on='card_id')
    test = test.merge(hist_transt_amt_summ, how='left', on='card_id')
    del(hist_transt_amt_summ)
    
    new_transt_amt_summ = new_trans.groupby(['card_id']).agg({c : ['count', 'sum', 'mean', 'max', 'std']}).reset_index()
    new_transt_amt_summ.columns = ['card_id', 'new_'+ c +'_cnt', 'new_'+ c +'_sum', 
                                    'new_'+ c +'_mean',  'new_'+ c +'_max', 'new_'+ c +'_std']
    train = train.merge(new_transt_amt_summ, how='left', on='card_id')
    test = test.merge(new_transt_amt_summ, how='left', on='card_id')
    del(new_transt_amt_summ)


# In[ ]:


#train.sort_values(by='target', ascending=True)
#C_ID_a4e600deef, C_ID_7e285a535a
train[train.card_id.isin(['C_ID_a4e600deef', 'C_ID_7e285a535a'])].head().T


# In[ ]:


ggplot(train, aes(x='hist_authorized_flag_mean', y='target')) +geom_point(color='red', alpha=0.4, size=0.2)


# In[ ]:


ggplot(train[train.hist_purchase_amount_sum<2000000], aes(x='hist_purchase_amount_sum', y='target')) +geom_point(color='red', alpha=0.4, size=0.2)


# In[ ]:


train_corr = train.corr()


# In[ ]:


plt.figure(figsize=(8,13))
train_corr['target'] = train_corr['target'].abs().sort_values(ascending=False)
train_corr.drop('target', axis=0)['target'].plot(kind='barh')
plt.show()


# Creating validation set for cross validation. 
# * train set size : 80%
# * validation set size : 20%

# In[ ]:


train = train.fillna(0)
test = test.fillna(0)


# In[ ]:


X_train = train.drop(['target', 'first_active_month', 'card_id'], axis=1, inplace=False)
y_train = train.target
X_test = test.drop([ 'first_active_month', 'card_id'], axis=1, inplace=False)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


print(X_train.shape, X_val.shape, X_test.shape)


# Let's run few algorithams with default/my favorite parameters
# 
# **Linear Regression:**

# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred)) #3.8346


# Decision Tree Regression:
# Taking max depth value as 5 and minimum sample leaf as 10. I almost always choose this parameter for first traila with decision tree based on my past experience.

# In[ ]:


dtr = DecisionTreeRegressor(max_depth = 6, min_samples_leaf=10)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred)) # 3.8272


# **KNN Regression:**

# In[ ]:


'''for i in range(1,30):
    knn = KNeighborsRegressor(n_neighbors=i, n_jobs=12)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(i, np.sqrt(mean_squared_error(y_test, y_pred)) )
'''    


# **Random Forest Regression:**
# 
# Again with some of my favorite parameters for first submission

# In[ ]:


'''rfr = RandomForestRegressor(max_depth=6, n_estimators=150, min_samples_leaf=5)
rfr.fit(X_train, y_train)
rf_pred = rfr.predict(X_val)
rf_train_pred = rfr.predict(X_train)
print(np.sqrt(mean_squared_error(y_val, rf_pred))) # 3.8027 #3.854 on LB
print(np.sqrt(mean_squared_error(y_train, rf_train_pred))) # 3.7342
'''


# In[ ]:


train_columns = X_train.columns
lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(train_columns))
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=list(train_columns))


# In[ ]:


params = {
    'objective': 'regression_l2',
    'metric': { 'rmse'},
    'num_leaves': 2000,
    'learning_rate': 0.1,
    'feature_fraction': 1,
    'verbose': 0,
    'max_depth' : 6,
    'min_data_in_leaf' : 8
}


# In[ ]:


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50) #3.76446


# In[ ]:


test_lgbm_pred = gbm.predict(X_test, num_iteration = gbm.best_iteration)


# In[ ]:


submission = pd.DataFrame(test_lgbm_pred, index=X_test.index)
submission.columns=['target']
submission = pd.concat([test.card_id, submission], axis=1)
submission.to_csv('submission_lgbm.csv', index=False)


# In[ ]:


lgb_fi = pd.DataFrame(gbm.feature_importance(), index=train_columns).reset_index()
lgb_fi.columns = ['column','score']


# In[ ]:


plotnine.options.figure_size = (14, 20)
ggplot(lgb_fi, aes(x='column', y='score')) +geom_bar(stat='identity') +theme(axis_text_x=element_text(rotation=90, hjust=1)) +plotnine.coord_flip()


# In[ ]:


lgb_fi.sort_values(by='score', ascending=False)


# In[ ]:




