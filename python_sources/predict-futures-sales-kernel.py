#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
from math import sqrt
from numpy import loadtxt
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")


# In[ ]:


items = pd.read_csv('input/items.csv')
shops = pd.read_csv('input/shops.csv')
cats = pd.read_csv('input/item_categories.csv')
train = pd.read_csv('input/sales_train_v2.csv')
test  = pd.read_csv('input/test.csv').set_index('ID')


# In[ ]:


plt.figure(figsize=(10,4))
plt.xlim(-100, train.item_cnt_day.max()*1.1)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)


# In[ ]:


train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]


# In[ ]:


plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), 0)
sns.boxplot(x=train.item_price)


# In[ ]:


median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median
train.head()


# In[ ]:


matrix = []
cols = ['date_block_num','shop_id', 'item_id']
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
#matrix.head()


# In[ ]:


matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix.drop(['item_name'], axis=1matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix.drop(['item_name'], axis=1, inplace=True)
#matrix.head(), inplace=True)
#matrix.head()


# In[ ]:


group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month'] .fillna(0) .clip(0,20) .astype(np.float16))
matrix.tail()


# In[ ]:


test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
#test.head()


# In[ ]:


matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)


# In[ ]:


def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df


# In[ ]:


matrix = lag_feature(matrix, [1,2], 'item_cnt_month')
matrix.head()


# In[ ]:


group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1, 2], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
matrix.tail()


# In[ ]:


group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
matrix.tail()


# In[ ]:


group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
matrix.tail()


# In[ ]:


group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_cat_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1, 2], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
matrix.tail()


# In[ ]:


matrix['month'] = matrix['date_block_num'] % 12


# In[ ]:


matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')


# In[ ]:


matrix = matrix[matrix.date_block_num > 1]
matrix.head()


# In[ ]:


def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
    return df
matrix = fill_na(matrix)
matrix.head()


# In[ ]:


import pickle
import gc
matrix.to_pickle('data.pkl')
del matrix
#del cache
del group
del items
del shops
del cats
del train
gc.collect();


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor, plot_importance
from xgboost import XGBRegressor, plot_importance
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


test  = pd.read_csv('input/test.csv').set_index('ID')
data = pd.read_pickle('data.pkl')
data.head()


# In[ ]:


X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# In[ ]:


gc.collect();


# In[ ]:


XGBmodel = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)


# In[ ]:


XGBmodel.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)


# In[ ]:


plot_importance(XGBmodel)


# In[ ]:


LGBMmodel=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=32,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.04,
        reg_lambda=0.073,
        min_split_gain=0.0222415,
        min_child_weight=40)


# In[ ]:


LGBMmodel.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)


# In[ ]:


plot_importance(LGBMmodel)


# In[ ]:


CATBmodel = CatBoostRegressor(
    iterations=100,
    random_seed=0,
    learning_rate=0.1,
    max_ctr_complexity=3, # To enable feature interactions
    has_time=True, # To disable random permutations
    boosting_type='Ordered', # To reduce overfitting
    loss_function='RMSE',
    od_type='Iter', 
    od_wait=10, # Early stopping
)


# In[ ]:


CATBmodel.fit(
    X_train, 
    Y_train, 
    #eval_metric="rmse", 
    eval_set=(X_valid, Y_valid), 
    verbose=True, 
    early_stopping_rounds = 10
)


# In[ ]:


xgb_Y_pred = XGBmodel.predict(X_valid).clip(0, 20)
xgb_Y_test = XGBmodel.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": xgb_Y_test
})
submission.to_csv('xgb_submission.csv', index=False)

pickle.dump(xgb_Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(xgb_Y_test, open('xgb_test.pickle', 'wb'))


# In[ ]:


cat_Y_pred = CATBmodel.predict(X_valid).clip(0, 20)
cat_Y_test = CATBmodel.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": cat_Y_test
})
submission.to_csv('catb_submission.csv', index=False)

pickle.dump(cat_Y_pred, open('catb_train.pickle', 'wb'))
pickle.dump(cat_Y_test, open('catb_test.pickle', 'wb'))


# In[ ]:


lgbm_Y_pred = LGBMmodel.predict(X_valid).clip(0, 20)
lgbm_Y_test = LGBMmodel.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": lgbm_Y_test
})
submission.to_csv('lgbm_submission.csv', index=False)

pickle.dump(cat_Y_pred, open('lgbm_train.pickle', 'wb'))
pickle.dump(cat_Y_test, open('lgbm_test.pickle', 'wb'))


# In[ ]:


df_mean_analysis = pd.DataFrame()
df_mean_analysis['lgbm'] = lgbm_Y_pred
df_mean_analysis['catb'] = cat_Y_pred
df_mean_analysis['xgb'] = xgb_Y_pred
df_mean_analysis.head()


# In[ ]:


XGBmodel.fit(df_mean_analysis, Y_valid)


# In[ ]:


import numpy as np
import pandas as pd
from xgboost import XGBRegressor, plot_importance
xgb_val = pd.read_pickle('xgb_train.pickle')
catb_val = pd.read_pickle('catb_train.pickle')
lgbm_val = pd.read_pickle('lgbm_train.pickle')
xgb_test = pd.read_pickle('xgb_test.pickle')
catb_test = pd.read_pickle('catb_test.pickle')
lgbm_test = pd.read_pickle('lgbm_test.pickle')


# In[ ]:


first_level = pd.DataFrame()
first_level['xgbm'] = xgb_val
first_level['catbm'] = catb_val
first_level['lgbm'] = lgbm_val
first_level.head()


# In[ ]:


first_level_test = pd.DataFrame()
first_level_test['xgbm'] = xgb_test
first_level_test['catbm'] = catb_test
first_level_test['lgbm'] = lgbm_test
first_level_test.head()


# In[ ]:


data = pd.read_pickle('data.pkl')
Y_valid = data[data.date_block_num == 33]['item_cnt_month']


# In[ ]:


from sklearn.model_selection import KFold, cross_val_predict
ensembler = XGBRegressor(
    max_depth=2,
    n_estimators=150,
    min_child_weight=100, 
    colsample_bytree=0.75, 
    subsample=0.75, 
    eta=0.1,    
    seed=42)


# In[ ]:


kfold = KFold(n_splits=5, random_state=42)
Y_pred = cross_val_predict(ensembler, first_level, Y_valid, cv=kfold)


# In[ ]:


from sklearn.metrics import mean_squared_error
print('The cross validation RMSE is', np.sqrt(mean_squared_error(Y_valid, Y_pred)))


# In[ ]:


ensembler.fit(first_level, Y_valid)


# In[ ]:


plot_importance(ensembler)


# In[ ]:


ensemble_pred = ensembler.predict(first_level_test)
#print('Train RMSE is', np.sqrt(mean_squared_error(ensemble_pred, Y_valid)))


# In[ ]:


test = pd.read_csv('input/test.csv')
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": ensemble_pred
})
submission.to_csv('submission.csv', index=False)

