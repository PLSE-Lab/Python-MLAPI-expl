#!/usr/bin/env python
# coding: utf-8

# # Starter Notebook XGBoost + EDA of Elo Merchant Data

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import seaborn as sns
import sklearn
import xgboost as xgb
plt.style.use('ggplot') # Lets make our plots pretty


# In[ ]:


# Read in the dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merch = pd.read_csv('../input/merchants.csv')
ht = pd.read_csv('../input/historical_transactions.csv')
ss = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


# Print data shapes
print('train shape', train.shape)
print('test shape', test.shape)
print('merchants shape', merch.shape)
print('sample submission shape', ss.shape)
print('historical_transactions', ht.shape)


# # What files do I need?
# You will need, at a minimum, the `train.csv` and `test.csv` files. These contain the card_ids that we'll be using for training and prediction.
# 
# `train.csv` and `test.csv` contain `card_ids` and information about the card itself - the first month the card was active, etc. train.csv also contains the target.

# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Plot Features
# There are three given features:
# - Feature 1 has 5 possible values
# - Feature 2 has 3 possible values
# - Feature 3 has 2 possible values
# 
# Train and test set appear so have similar distribution

# In[ ]:


features = ['feature_1', 'feature_2', 'feature_3']
for feature in features:
    fig, axes = plt.subplots(nrows=1, ncols=2)
    train.groupby(feature).count()['first_active_month'].plot(kind='bar',
                                                              title='train {}'.format(feature),
                                                              figsize=(15, 4),
                                                              ax=axes[0])
    test.groupby(feature).count()['first_active_month'].plot(kind='bar',
                                                              title='test {}'.format(feature),
                                                              figsize=(15, 4))
    plt.show()


# # Plot Target Variable
# - The target variable is normally distributed around zero
# - The exception being some very low values below -30
# - Possibly identify these low values and remove them from the training?
# - Still need to transform the target to make it more normally distributed ?

# In[ ]:


train['target'].plot(kind='hist', bins=50, figsize=(15, 5), title='Target variable distribution')
plt.show()


# In[ ]:


train['target_log5p'] = (train['target'] + 5).apply(np.log1p)
train['target_log5p'].plot(kind='hist', bins=50, figsize=(15, 5), title='Target variable log+5 transform distribution')
plt.show()


# # Plot the first_active_month - train and test

# In[ ]:


train['first_active_month'] = pd.to_datetime(train['first_active_month'])
train.groupby('first_active_month').count()['card_id'].plot(figsize=(15,5),
                                                            title='Count of First Active Month in Train Set',
                                                           color='r')
plt.show()


# In[ ]:


test['first_active_month'] = pd.to_datetime(test['first_active_month'])
test.groupby('first_active_month').count()['card_id'].plot(figsize=(15,5),
                                                           title='Count of First Active Month in Test Set',
                                                          color='b')
plt.show()


# # Historical Transactions
# `historical_transactions.csv` contains up to 3 months' worth of transactions for every card at any of the provided merchant_ids.

# In[ ]:


ht.head()


# # Create Features and bare bones first attempt XGBoost model

# In[ ]:


# Create features from historic transactions
gdf = ht.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]

def merge_hist_transactions(df, hist_agg_df):
    """Merges the historic transactions data with train or test set"""
    return pd.merge(df, hist_agg_df, on="card_id", how="left")


# In[ ]:


def create_features(df, target=False):
    """
    Creates raw features including one hot encoding for train and test
    """
    # One-hot encode features
    feat1 = pd.get_dummies(df['feature_1'], prefix='f1_')
    feat2 = pd.get_dummies(df['feature_2'], prefix='f2_')
    feat3 = pd.get_dummies(df['feature_3'], prefix='f3_')
    # Numerical representation of the first active month
    fam_num = pd.to_timedelta(df['first_active_month']).dt.total_seconds().astype(int)
    # Historical features
    hist_features = df[['sum_hist_trans','mean_hist_trans','std_hist_trans','min_hist_trans','max_hist_trans']]
    if target:
        return pd.concat([feat1, feat2, feat3, fam_num, hist_features], axis=1, sort=False), df['target']
    return pd.concat([feat1, feat2, feat3, fam_num, hist_features], axis=1, sort=False)


# # Na Values
# We have one NA value in our test set for the first active month

# In[ ]:


# Fill the NA value with the last month?
test['first_active_month'] = pd.to_datetime(test['first_active_month'].fillna('2018-01-01'))


# In[ ]:


train_with_hist = merge_hist_transactions(train, gdf)
test_with_hist = merge_hist_transactions(test, gdf)


# In[ ]:


X_train, y_train = create_features(train_with_hist, target=True)
X_test = create_features(test_with_hist, target=False)


# # Train model

# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train.values)
model = xgb.train(params={'silent':1}, dtrain=dtrain, verbose_eval=False, num_boost_round=100)


# In[ ]:


# Predict and format submission
dtest = xgb.DMatrix(X_test)
preds = model.predict(dtest)
# Our submission
submission = pd.concat([test['card_id'], pd.Series(preds)], axis = 1)
submission = submission.rename(columns={0:'target'})
print('submission shape', submission.shape)


# In[ ]:


submission.head()


# In[ ]:


# Save our output for submission
submission.to_csv('submission.csv', header=True, index=False)


# # Train model removing <-30 targets from training
# Just as a test lets try removing these values from our training

# In[ ]:


train_remneg30 = train_with_hist.loc[train_with_hist['target'] > -30]
X_train_rem30, y_train_rem30 = create_features(train_remneg30, target=True)
# X_test = create_features(test, target=False)
dtrain = xgb.DMatrix(X_train_rem30, label=y_train_rem30.values)
model2 = xgb.train(params={'silent':1}, dtrain=dtrain, verbose_eval=False, num_boost_round=100)
# Predict and format submission
dtest = xgb.DMatrix(X_test)
preds = model2.predict(dtest)
# Our submission
submission2 = pd.concat([test['card_id'], pd.Series(preds)], axis = 1)
submission2 = submission2.rename(columns={0:'target'})
submission2.to_csv('submission_remneg30.csv', header=True, index=False)


# # Compare the two submission distributions

# In[ ]:


submission2['target'].plot(kind='hist', bins=50, title='submission removing <-30 values')
plt.show()


# In[ ]:


submission['target'].plot(kind='hist', bins=50, title='submission not removing <-30 values')
plt.show()
plt.show()


# # Still to come.....
# 1. Explore the below additional data
# 2. Create a more complex model using historical transactions
# 3. Utilize new merchant transactions
# 
# ## The `historical_transactions.csv` and `new_merchant_transactions.csv` files contain information about each card's transactions.
# 
# ## `historical_transactions.csv` contains up to 3 months' worth of transactions for every card at any of the provided merchant_ids.
# 
# ## `new_merchant_transactions.csv` contains the transactions at new merchants (merchant_ids that this particular card_id has not yet visited) over a period of two months.
# 
# ## `merchants.csv` contains aggregate information for each merchant_id represented in the data set.
# 

# In[ ]:




