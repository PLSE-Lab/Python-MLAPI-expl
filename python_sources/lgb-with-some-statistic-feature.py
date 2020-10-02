#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import numpy as np
from multiprocessing import Pool
import datetime
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import warnings
from xgboost import XGBRegressor
from joblib import Parallel,delayed
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train['first_active_month'] = pd.to_datetime(train['first_active_month'] )
train['year'] = pd.to_datetime(train['first_active_month']).dt.year
train['month'] = pd.to_datetime(train['first_active_month']).dt.month


# In[ ]:


test['first_active_month'] = pd.to_datetime(test['first_active_month'])
test['year'] = pd.to_datetime(test['first_active_month']).dt.year
test['month'] = pd.to_datetime(test['first_active_month']).dt.month


# In[ ]:


ht = pd.read_csv('../input/historical_transactions.csv')
nmt = pd.read_csv('../input/new_merchant_transactions.csv')
merchants = pd.read_csv('../input/merchants.csv')


# In[ ]:


ht['purchase_date'] = pd.to_datetime(ht['purchase_date'])
nmt['purchase_date'] = pd.to_datetime(nmt['purchase_date'])


# In[ ]:


ht['category_2'] = ht['category_2'].fillna(0)
ht['category_3'] = ht['category_3'].fillna('E')


# In[ ]:


train['active_till_now'] = train['first_active_month'].map(lambda x:(datetime.datetime.today() - x).days)
test['active_till_now'] = test['first_active_month'].map(lambda x:(datetime.datetime.today() - x).days)


# In[ ]:


train['target'].hist(bins=20)


# In[ ]:


card_id_cnt = ht.groupby('card_id')['city_id'].count().reset_index()
card_id_cnt.columns = ['card_id','purchase_cnt']


# In[ ]:


train = pd.merge(train,card_id_cnt,on='card_id',how='left')
test = pd.merge(test,card_id_cnt,on='card_id',how='left')


# In[ ]:


amount = ht.groupby('card_id').agg({'purchase_amount':['mean','std','max']})
amount.columns = ['amount_mean','amount_std','amount_max']
amount = amount.reset_index()


# In[ ]:


month_lag = ht.groupby('card_id').agg({'month_lag':['max','min','std','mean']})
month_lag.columns = ['lag_max','lag_min','lag_std','lag_mean']
month_lag = month_lag.reset_index()


# In[ ]:


def get_stat_feature(df):
    city_cnt = df['city_id'].nunique()
    mer_cnt = df['merchant_id'].nunique()
    state_cnt = df['state_id'].nunique()
    installments = len(df['installments']>0)
    avg_installments = df['installments'].mean()
    max_installments = df['installments'].max()
    newly_date = df['purchase_date'].max()
    last_buy = (datetime.datetime.today() - newly_date).days
    return [df['card_id'].values[0],city_cnt,mer_cnt,state_cnt,installments,avg_installments,max_installments,last_buy]
def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=2)(delayed(func)(group) for name, group in dfGrouped)
    return retLst
ht_feature = ht.groupby('card_id')
ht_res = applyParallel(ht_feature,get_stat_feature)


# In[ ]:


stat_df = pd.DataFrame(ht_res,columns=['card_id','city_cnt','mer_cnt','state_cnt','installments','avg_installments',
                                       'max_installments','last_buy'])


# In[ ]:


train = pd.merge(train,amount,on='card_id',how='left')
test = pd.merge(test,amount,on='card_id',how='left')


# In[ ]:


train = pd.merge(train,month_lag,on='card_id',how='left')
test = pd.merge(test,month_lag,on='card_id',how='left')


# In[ ]:


train = pd.merge(train,stat_df,on='card_id',how='left')
test = pd.merge(test,stat_df,on='card_id',how='left')


# In[ ]:


cat_feature = ['feature_1','feature_2','feature_3']
num_feature = ['active_till_now','purchase_cnt','amount_mean','amount_std','amount_max','lag_max','lag_min','lag_std','lag_mean',
              'city_cnt','mer_cnt','state_cnt','installments','avg_installments','max_installments','last_buy']


# In[ ]:


for cat in cat_feature:
    train[cat] = train[cat].map(lambda x:str(x))
    test[cat] = test[cat].map(lambda x:str(x))


# In[ ]:


train_cat_feature = pd.get_dummies(train[cat_feature])
test_cat_feature = pd.get_dummies(test[cat_feature])


# In[ ]:


ss = StandardScaler()
train_num_feature = ss.fit_transform(train[num_feature])
test_num_feature = ss.transform(test[num_feature])


# In[ ]:


train_feature = np.hstack([train_num_feature,train_cat_feature.values])
test_feature = np.hstack([test_num_feature,test_cat_feature.values])


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[ ]:


res = []
feature_imp = []
for tr,va in KFold(n_splits=10,random_state=2018).split(train_feature,train['target'].values):
    lgbmr = LGBMRegressor(num_leaves=16,n_estimators=50)
    lgbmr.fit(train_feature[tr],train['target'].values[tr],
              eval_set=(train_feature[va],train['target'].values[va]),
              eval_metric='rmse',
              verbose=50)
    feature_imp.append(lgbmr.feature_importances_)
    res.append(lgbmr.predict(test_feature))


# In[ ]:


f = np.mean(feature_imp,axis=0)


# In[ ]:


list(zip(cat_feature+num_feature,f))


# In[ ]:


res_lr = []
feature_imp = []
for tr,va in KFold(n_splits=10,random_state=2018).split(train_feature,train['target'].values):
    lr = LinearRegression(n_jobs=-1)
    lr.fit(train_feature[tr],train['target'].values[tr])
    print(np.sqrt(mean_squared_error(train['target'].values[va],lr.predict(train_feature[va]))))


# In[ ]:


avg_res = np.mean(res,axis=0)
test['target'] = avg_res
test[['card_id','target']].to_csv('predict.csv',index=False)

