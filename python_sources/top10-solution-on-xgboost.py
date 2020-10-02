#!/usr/bin/env python
# coding: utf-8

# # Load the data

# In[6]:


import pandas as pd
import numpy as np
import gc
import itertools
from tqdm import tqdm_notebook
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
DATA_FOLDER = '../input/'
train     = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
items     = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_cats = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops     = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
test      = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
test['date_block_num'] = 34
groupby_cols = ['date_block_num', 'shop_id', 'item_id']


# # Add some features
# 
# I did not extract too much information from text fields, just 2 features:
# **city** (which is also encoded in **shop_id**),
# and **meta_cat** (some meta category, higher level aggregation of a **category** field)

# In[2]:


#category
cat = items[['item_id', 'item_category_id']].drop_duplicates()
cat.set_index(['item_id'], inplace=True)
cat = cat.item_category_id
train['cat'] = train.item_id.map(cat)

#meta_category   games, gifts, work, etc.
item_cats['meta_cat'] = item_cats.item_category_name.apply(lambda x: x.split(' ')[0])
item_cats['meta_cat'] = pd.Categorical(item_cats.meta_cat).codes
item_cats.set_index(['item_category_id'], inplace=True)
meta_cat = item_cats.meta_cat
train['meta_cat'] = train.cat.map(meta_cat)

#city
shops['city'] = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
shops['city'] = pd.Categorical(shops['city']).codes
city = shops.city
train['city'] = train.shop_id.map(city)

#year
year = pd.concat([train.date_block_num, train.date.apply(lambda x: int(x.split('.')[2]))], axis=1).drop_duplicates()
year.set_index(['date_block_num'], inplace=True)
year = year.date.append( pd.Series([2015], index=[34]) )

#month
month = pd.concat([train.date_block_num, train.date.apply(lambda x: int(x.split('.')[1]))], axis=1).drop_duplicates()
month.set_index(['date_block_num'], inplace=True)
month = month.date.append( pd.Series([11], index=[34]) )


# In[3]:


all_shops_items = []

for block_num in train['date_block_num'].unique():
    unique_shops = train[ train['date_block_num']==block_num ]['shop_id'].unique()
    unique_items = train[ train['date_block_num']==block_num ]['item_id'].unique()
    all_shops_items.append( np.array( list( itertools.product([block_num], unique_shops, unique_items) ), dtype='int32' ) )

df = pd.DataFrame( np.vstack( all_shops_items ), columns=groupby_cols , dtype='int32')
df = df.append(test )

df['ID'] = df.ID.fillna(-1).astype('int32')

df['year'] = df.date_block_num.map(year)
df['month'] = df.date_block_num.map(month)
df['cat'] = df.item_id.map(cat)
df['meta_cat'] = df.cat.map(meta_cat)
df['city'] = df.shop_id.map(city)
train['cat'] = train.item_id.map(cat)


# In[4]:


df.sample(5)


# # Some plots.
# Overall trend - the sales are decreasing. Peaks on December. date_block_num's [0: 12] contain quite a lot of training data, it is useful to keep them in the training set.

# In[5]:


gb = train.groupby(by=['date_block_num'], as_index=False).agg({ 'item_cnt_day': ['sum']})
sns.barplot(x=gb['date_block_num'], y=gb['item_cnt_day']['sum'])


# Most of the target values are just Zeros.

# In[6]:


sns.distplot(train['item_cnt_day'])


# Zeros and ones are just dominating.

# In[12]:


sns.distplot( train[ train['item_cnt_day'] > 0 ]['item_cnt_day'].clip(0, 20) )


# # Calculating different aggregations

# In[5]:


gb = train.groupby(by=groupby_cols, as_index=False).agg({ 'item_cnt_day': ['sum']})
gb.columns = [val[0] if val[-1]=='' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum':'target'}, inplace=True)
df = pd.merge(df, gb, how='left', on=groupby_cols)

gb = train.groupby(by=['date_block_num', 'item_id'], as_index=False).agg({'item_cnt_day':['sum']})
gb.columns = [val[0] if val[-1]=='' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum':'target_item'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'item_id'])

gb = train.groupby(by=['date_block_num', 'shop_id'], as_index=False).agg({'item_cnt_day':['sum']})
gb.columns = [val[0] if val[-1]=='' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum':'target_shop'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'shop_id'])

gb = train.groupby( by=['date_block_num', 'cat'], as_index=False).agg({ 'item_cnt_day': ['sum'] })
gb.columns = [val[0] if val[-1]=='' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum':'target_cat'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'cat'])

gb = train.groupby(by=['date_block_num', 'item_id'], as_index=False).agg({'item_price':['mean', 'max']})
gb.columns = [val[0] if val[-1]=='' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_price_mean':'target_price_mean', 'item_price_max':'target_price_max'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'item_id'])


# # Clipping some outlier prices

# In[10]:


df['target_price_mean'] = np.minimum(df['target_price_mean'], df['target_price_mean'].quantile(0.99))
df['target_price_max'] = np.minimum(df['target_price_max'], df['target_price_max'].quantile(0.99))

df.fillna(0, inplace=True)
df['target'] = df['target'].clip(0, 20)
df['target_zero'] = (df['target'] > 0).astype("int32")


# # Mean Encodings

# In[11]:


get_ipython().run_cell_magic('time', '', "for enc_cols in [['shop_id', 'cat'], ['shop_id', 'item_id'], ['shop_id'], ['item_id']]:\n    \n    col = '_'.join(['enc', *enc_cols])\n    col2 = '_'.join(['enc_max', *enc_cols])\n    df[col] = np.nan\n    df[col2] = np.nan\n    \n    print(col, col2)\n    \n    for d in tqdm_notebook(df.date_block_num.unique()):\n        \n\n        f1 = df.date_block_num < d\n        f2 = df.date_block_num == d\n\n        gb = df.loc[f1].groupby(enc_cols)[['target']].mean().reset_index()\n        enc = df.loc[f2][enc_cols].merge(gb, on=enc_cols, how='left' )[['target']].copy()\n        enc.set_index(df.loc[f2].index, inplace=True)\n        df.loc[f2, col] = enc['target']\n        \n        gb = df.loc[f1].groupby(enc_cols)[['target']].max().reset_index()\n        enc = df.loc[f2][enc_cols].merge(gb, on=enc_cols, how='left' )[['target']].copy()\n        enc.set_index(df.loc[f2].index, inplace=True)\n        df.loc[f2, col2] = enc['target']\n        ")


# In[14]:


def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float32_cols = [c for c in df if df[c].dtype == "float64"]
    int32_cols = [c for c in df if df[c].dtype in ["int64", "int16", "int8"] ]
            
    # Downcast
    df[float32_cols] = df[float32_cols].astype(np.float32)
    df[int32_cols]   = df[int32_cols].astype(np.int32)
        
    return df


# In[15]:


df.fillna(0, inplace=True)
df = downcast_dtypes(df)


# # Historical values

# In[16]:


import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import os
import gc
from time import sleep

groupby_cols = ['date_block_num', 'shop_id', 'item_id']
shift_range = [1, 2, 3, 4, 5, 12]


# In[17]:


get_ipython().run_cell_magic('time', '', "\nshifted_columns = [c for c in df if 'target' in c]\nprint(shifted_columns)\n\n\nfor shift in tqdm_notebook(shift_range):\n    \n    \n    shifted_data = df[groupby_cols + shifted_columns].copy()\n    shifted_data['date_block_num'] = shifted_data['date_block_num'] + shift\n    \n    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in shifted_columns else x\n    shifted_data = shifted_data.rename(columns=foo)\n    \n    \n    df = pd.merge(df, shifted_data, how='left', on=groupby_cols).fillna(0)\n    df = downcast_dtypes(df)\n\n    del shifted_data\n    gc.collect()\n    sleep(1)")


# # Features Interaction
# A big breakthrough happened, when i encoded these interactions. Check it out.

# In[18]:


df['target_trend_1_2'] = df['target_lag_1'] - df['target_lag_2']
df['target_predict_1_2'] = df['target_lag_1']*2 - df['target_lag_2']

df['target_trend_3_4'] = df['target_lag_1'] + df['target_lag_2'] - df['target_lag_3'] - df['target_lag_4']
df['target_predict_3_4'] = (df['target_lag_1'] + df['target_lag_2'])*2 - df['target_lag_3'] - df['target_lag_4']

df['target_item_trend_1_2'] = df['target_item_lag_1'] - df['target_item_lag_2']
df['target_item_trend_3_4'] = df['target_item_lag_1'] + df['target_item_lag_2'] - df['target_item_lag_3'] - df['target_item_lag_4']
df['target_shop_trend_1_2'] = df['target_shop_lag_1'] - df['target_shop_lag_2']
df['target_shop_trend_3_4'] = df['target_shop_lag_1'] + df['target_shop_lag_2'] - df['target_shop_lag_3'] - df['target_shop_lag_4']


# In[19]:


df = downcast_dtypes(df)
df.to_pickle("df.pkl")


# # XGBoost part

# In[20]:


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_curve, roc_auc_score
import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


df = pd.read_pickle("df.pkl")

drop_columns = [c for c in df if c[-1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] and c.startswith('target')]
drop_columns += ['ID']
features = df.columns.difference( drop_columns )

f0 = df.date_block_num < 34
f1 = df.date_block_num == 34

train, val = train_test_split(df[f0], test_size=0.2, stratify=df[f0]['target'])
test = df[f1]


# In[24]:


Train = xgb.DMatrix( train[features], train['target'] )
Val = xgb.DMatrix( val[features], val['target'] )
Test = xgb.DMatrix( test[features])


# In[25]:


del df
gc.collect()


# For the hyperparameters optimization see the Hyperopt section below. Here are the parameters, which turned out to be optimal.

# In[26]:


get_ipython().run_cell_magic('time', '', "\nxgb_params = {'booster': 'gbtree',\n          'eta': .1,\n          'min_child_weight': 100,\n          'max_depth': 6,\n          'objective': 'reg:linear',\n          'eval_metric': 'rmse',\n          'silent': False,\n          'nthread': 8}\n\nmodel = xgb.train(xgb_params, Train, 1000, [(Train, 'Train'), (Val, 'Val')], early_stopping_rounds=10, verbose_eval=1 )")


# In[27]:


test['item_cnt_month'] = model.predict(Test).clip(0, 20)
test[['ID', 'item_cnt_month']].sort_values('ID').to_csv('boost_pred.csv', index=False)


# In[12]:


#pickle.dump(model, open("xgb.pickle", "wb"))


# In[31]:


#model = pickle.load(open("xgb.pickle", "rb"))


# **item_id**, **date_block_num** and **month** have quite a predictive power :)

# In[32]:


xgb.plot_importance(booster=model, max_num_features=12)


# # HyperOpt

# In[1]:


from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import gc


# In[2]:


df = pd.read_pickle("df.pkl")

drop_columns = [c for c in df if c[-1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] and c.startswith('target')]
drop_columns += ['item_id', 'shop_id']
drop_columns += ['ID']

features = df.columns.difference( drop_columns )

f0 = df.date_block_num < 34
f1 = df.date_block_num == 34


train, val = train_test_split(df[f0], test_size=0.1, stratify=df[f0]['target'])
test = df[f1]

Train = xgb.DMatrix( train[features], train['target'] )
Val = xgb.DMatrix( val[features], val['target'] )
Test = xgb.DMatrix( test[features])

del df
gc.collect()


# In[3]:


def objective(params):
    
    params = {
        'booster': 'gbtree',
        'silent': True,
        'nthread': 8,
        
        'eta': "{:.3f}".format(params['eta']),
        'gamma': "{:.3f}".format(params['gamma']),
        'max_depth': int(params['max_depth']),
        'min_child_weight': int(params['min_child_weight']),
        'subsample': "{:.3f}".format(params['subsample']),        
        'colsample_bytree': "{:.3f}".format(params['colsample_bytree']),
        'lambda': "{:.3f}".format(params['lambda']),
        'alpha': "{:.3f}".format(params['alpha']),
          
        'objective': 'reg:linear',
        'eval_metric': 'rmse'
        
    }

    watchlist = [(Train, 'Train'), (Val, 'Val')]
    
    model = xgb.train(
        params,
        Train,
        25,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=100 )
    
    

    y_pred = model.predict(Val)
    score = np.sqrt( mean_squared_error(val['target'].clip(0, 20), y_pred.clip(0, 20)))
    
    #score = 1
                    
    print("score {:.3f} params {}".format(score, params))
    return score

space = {
    'eta': hp.uniform('learning_rate', 0.01, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),'gamma': hp.uniform('gamma', 0.0, 0.3),
    'max_depth': hp.quniform('max_depth', 2, 12, 1),
    'min_child_weight': hp.uniform('min_child_weight', 1, 200),
    'subsample': hp.uniform('subsample', 0.5, 1.0),    
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'lambda': hp.uniform('lambda', 0.0, 0.3),
    'alpha': hp.uniform('alpha', 0.0, 0.3),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=3)

print(best)


# # Ensembling

# I had one xgboost solution for 0.93, this one i've got training on  clip(0, 40) training set.
# Another solution for 0.91, where training set was clipped to(0, 20)
# Combinig those 2 solutions gave quite a good result of 0.90

# At the moment this final ensemble solution scored 0.904 on public leaderboard.
# ![top6.jpg](attachment:top6.jpg)

# In[ ]:




