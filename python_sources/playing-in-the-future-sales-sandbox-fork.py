#!/usr/bin/env python
# coding: utf-8

# **Loading Libraries and Data**

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime


# In[ ]:


from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
# items = pd.read_csv('../input/items.csv',encoding ='ISO-8859-1')
# item_cats = pd.read_csv('../input/item_categories.csv',encoding ='ISO-8859-1')
# shops = pd.read_csv('../input/shops.csv',encoding ='ISO-8859-1')

print('train:', train.shape, 'test:', test.shape)


#  ## Clean data
# 
# ### Dangerous: Keep only item-store combinations that are present in the test data
# * + Makes training faster
# * - Misses out on event s caused by other items being sold at same time, and ability to learn broader historical patterns
# 
# ### Drop -1s'
# ### Drop duplicate rows

# In[ ]:


train = train.loc[train.item_cnt_day<=1001] # there is only 1 item
train = train.loc[train.item_price<100001]


# In[ ]:


# drop duplicates
subset = ['date','shop_id','item_id','item_cnt_day']
print(train.duplicated(subset=subset).value_counts())
train.drop_duplicates(subset=subset, inplace=True)


# In[ ]:


# drop shops&items not in test data
print('Orig train shape:', train.shape[0])
test_shops = test.shop_id.unique()
test_items = test.item_id.unique()
print(len(test_shops),"Test shops")
print(len(test_items),"Test items")

train = train[train.shop_id.isin(test_shops)]
train = train[train.item_id.isin(test_items)]

print('New train (intersecting test) shape:', train.shape)


# In[ ]:


## Do this transform only on monthly/target data. Leave it in for the daily data (elsewhere), useful for features
train = train.loc[train.item_cnt_day != -1]


# **Difference betwee train and test**

# In[ ]:


[c for c in train.columns if c not in test.columns]


# In[ ]:


train.head()


# **Adding Features**
# 
# * Text Features
# * Date Features (Not necessarily needed for monthly summary but may help if using daily preds)

# In[ ]:


# #Text Features
# feature_cnt = 5
# tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
# items['item_name_len'] = items['item_name'].map(len) #Lenth of Item Description
# items['item_name_wc'] = items['item_name'].map(lambda x: len(str(x).split(' '))) #Item Description Word Count
# txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
# cols = txtFeatures.columns
# for i in range(feature_cnt):
#     items['item_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
# items.head()


# In[ ]:


# #Text Features
# feature_cnt = 5
# tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
# item_cats['item_category_name_len'] = item_cats['item_category_name'].map(len)  #Lenth of Item Category Description
# item_cats['item_category_name_wc'] = item_cats['item_category_name'].map(lambda x: len(str(x).split(' '))) #Item Category Description Word Count
# txtFeatures = pd.DataFrame(tfidf.fit_transform(item_cats['item_category_name']).toarray())
# cols = txtFeatures.columns
# for i in range(feature_cnt):
#     item_cats['item_category_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
# item_cats.head()


# In[ ]:


# #Text Features
# feature_cnt = 5
# tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
# # shops['shop_name_len'] = shops['shop_name'].map(len)  #Lenth of Shop Name
# # shops['shop_name_wc'] = shops['shop_name'].map(lambda x: len(str(x).split(' '))) #Shop Name Word Count
# txtFeatures = pd.DataFrame(tfidf.fit_transform(shops['shop_name']).toarray())
# cols = txtFeatures.columns
# for i in range(feature_cnt):
#     shops['shop_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
# shops.head()


# In[ ]:


## From : https://www.kaggle.com/plasticgrammer/predict-future-sales-training

item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])
# item_grp = pd.Categorical(item_grp).codes
item_cats['item_group'] = item_grp
#item_cats = item_cats.join(pd.get_dummies(item_grp, prefix='item_group', drop_first=True))

items = pd.merge(items, item_cats.loc[:,['item_category_id','item_group']], on=['item_category_id'], how='left')

shops["city"] = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])


# ### Merge 

# In[ ]:


#Make Monthly
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
# train['year'] = train['date'].dt.year

# train = train.drop(['date','item_price'], axis=1)
train.drop(['item_price'], axis=1,inplace=True)

train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})
#Monthly Mean
shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
#Add Mean Feature
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])
#Last Month (Oct 2015)
shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})
shop_item_prev_month.head()
#Add Previous Month Feature
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id'])#.fillna(0.)
#Items features
train = pd.merge(train, items, how='left', on='item_id')
#Item Category features
train = pd.merge(train, item_cats, how='left', on='item_category_id')
#Shops features
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()


# In[ ]:


print(train.shape[0])
train.drop_duplicates(inplace=True)
print(train.shape[0])


# In[ ]:


test['month'] = 11
# test['year'] = 2015
test['date_block_num'] = 34
#Add Mean Feature
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])#.fillna(0.)
#Add Previous Month Feature
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id','item_id'])#.fillna(0.)
#Items features
test = pd.merge(test, items, how='left', on='item_id')
#Item Category features
test = pd.merge(test, item_cats, how='left', on='item_category_id')
#Shops features
test = pd.merge(test, shops, how='left', on='shop_id')
# test['item_cnt_month'] = 0.
test.head()


# In[ ]:


print(test.shape[0])
test.drop_duplicates(subset=["ID"],inplace=True)
print(test.shape[0])


# ## Extra stuff
# * add item-store key
# * add date for test so column will match
# * Section should be commented out if you want your ml to run 

# In[ ]:


train["shop_item"] = train["shop_id"].astype(str) + "-" + train["item_id"].astype(str)
test["shop_item"] = test["shop_id"].astype(str) + "-" + test["item_id"].astype(str)

test["date"] = datetime.date(2015, 11, 30)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


set(train.columns)-set(test.columns)


# In[ ]:


set(test.columns)-set(train.columns)


# ## Export

# In[ ]:


train.to_csv("train_monthly_sales.csv.gz",index=False,compression="gzip",encoding="utf8")
test.to_csv("test_monthly_sales.csv.gz",index=False,compression="gzip",encoding="utf8")


# **Visualize**

# In[ ]:


df_all = pd.concat((train, test), axis=0, ignore_index=True)
stores_hm = df_all.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
fig, ax = plt.subplots(figsize=(10,10))
_ = sns.heatmap(stores_hm, ax=ax, cbar=False)


# In[ ]:


# stores_hm = train.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
# fig, ax = plt.subplots(figsize=(10,10))
# _ = sns.heatmap(stores_hm, ax=ax, cbar=False)


# **Label Encoding**
# 
# * Try different approaches - weight based sequence, etc.

# In[ ]:


# for c in ['shop_name','item_name','item_category_name']:
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(train[c].unique())+list(test[c].unique()))
#     train[c] = lbl.transform(train[c].astype(str))
#     test[c] = lbl.transform(test[c].astype(str))
#     print(c)


# **Train & Predict Models**

# In[ ]:


# col = [c for c in train.columns if c not in ['item_cnt_month']]
# #Validation Hold Out Month
# x1 = train[train['date_block_num']<33]
# y1 = np.log1p(x1['item_cnt_month'].clip(0.,20.))
# x1 = x1[col]
# x2 = train[train['date_block_num']==33]
# y2 = np.log1p(x2['item_cnt_month'].clip(0.,20.))
# x2 = x2[col]

# reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=18)
# reg.fit(x1,y1)
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y2.clip(0.,20.),reg.predict(x2).clip(0.,20.))))
# #full train
# reg.fit(train[col],train['item_cnt_month'].clip(0.,20.))
# test['item_cnt_month'] = reg.predict(test[col]).clip(0.,20.)
# test[['ID','item_cnt_month']].to_csv('submission.csv', index=False)


# **Happy Kaggling :)**
# 
# * Try XGBoost, LightGBM, CatBoost next
# * Try some more Scikit-Learn Linear Regressors and more
# * Also try TensorFlow, Keras, PyTorch or other NN models for extra fun
# * Add more text features
# * Make some ensembles
# * Tune your models further
# * Add some awesome visualizations
# * Comment on Kernels for feedback, fork some, share your own versions
# 
# Have some fun!

# dd**Getting Started with More Models**
# 
# * Off to model tuning land you go now...

# In[ ]:


# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostRegressor
# from multiprocessing import *

# #XGBoost
# def xgb_rmse(preds, y):
#     y = y.get_label()
#     score = np.sqrt(metrics.mean_squared_error(y.clip(0.,20.), preds.clip(0.,20.)))
#     return 'RMSE', score

# params = {'eta': 0.2, 'max_depth': 4, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'seed': 18, 'silent': True}
# #watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
# #xgb_model = xgb.train(params, xgb.DMatrix(x1, y1), 100,  watchlist, verbose_eval=10, feval=xgb_rmse, maximize=False, early_stopping_rounds=20)
# #test['item_cnt_month'] = xgb_model.predict(xgb.DMatrix(test[col]), ntree_limit=xgb_model.best_ntree_limit)
# #test[['ID','item_cnt_month']].to_csv('xgb_submission.csv', index=False)

# #LightGBM
# def lgb_rmse(preds, y):
#     y = np.array(list(y.get_label()))
#     score = np.sqrt(metrics.mean_squared_error(y.clip(0.,20.), preds.clip(0.,20.)))
#     return 'RMSE', score, False

# params = {'learning_rate': 0.2, 'max_depth': 7, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'mse', 'is_training_metric': False, 'seed': 18}
# #lgb_model = lgb.train(params, lgb.Dataset(x1, label=y1), 100, lgb.Dataset(x2, label=y2), feval=lgb_rmse, verbose_eval=10, early_stopping_rounds=20)
# #test['item_cnt_month'] = lgb_model.predict(test[col], num_iteration=lgb_model.best_iteration)
# #test[['ID','item_cnt_month']].to_csv('lgb_submission.csv', index=False)

# #CatBoost
# cb_model = CatBoostRegressor(iterations=100, learning_rate=0.2, depth=7, loss_function='RMSE', eval_metric='RMSE', random_seed=18, od_type='Iter', od_wait=20) 
# cb_model.fit(x1, y1, eval_set=(x2, y2), use_best_model=True, verbose=False)
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y2.clip(0.,20.), cb_model.predict(x2).clip(0.,20.))))
# test['item_cnt_month'] += cb_model.predict(test[col])
# test['item_cnt_month'] /= 2
# test[['ID','item_cnt_month']].to_csv('cb_blend_submission.csv', index=False)


# In[ ]:


#test['item_cnt_month'] = np.expm1(test['item_cnt_month'])
#test[['ID','item_cnt_month']].to_csv('cb_submission_exp.csv', index=False)

