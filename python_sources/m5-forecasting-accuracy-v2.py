#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import random

import warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


warnings.filterwarnings(action='once')


# In[ ]:


calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], index_col='date')
calendar.head(5)


# In[ ]:


calendar.fillna('0', inplace=True)

label_encoder = LabelEncoder()
label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

# Apply label encoder 
for col in label_cols:
    calendar[col] = label_encoder.fit_transform(calendar[col])

calendar.head(5)


# In[ ]:


calendar['is_weekend'] = calendar['wday'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
seasons = {1: 1, 2: 1, 12: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4 }
calendar['season'] = calendar['month'].apply(lambda x: seasons[x])
calendar.head()


# In[ ]:


calendar.info()


# In[ ]:


calendar['wm_yr_wk'] = calendar['wm_yr_wk'].astype(np.int16)
calendar['wday'] = calendar['wday'].astype(np.int8)
calendar['month'] = calendar['month'].astype(np.int8)
calendar['year'] = calendar['year'].astype(np.int16)
calendar['snap_CA'] = calendar['snap_CA'].astype(np.int8)
calendar['snap_TX'] = calendar['snap_TX'].astype(np.int8)
calendar['snap_WI'] = calendar['snap_WI'].astype(np.int8)
calendar['is_weekend'] = calendar['is_weekend'].astype(np.int8)
calendar['season'] = calendar['season'].astype(np.int8)
calendar['event_name_1'] = calendar['event_name_1'].astype(np.int16)
calendar['event_type_1'] = calendar['event_type_1'].astype(np.int8)
calendar['event_name_2'] = calendar['event_name_2'].astype(np.int8)
calendar['event_type_2'] = calendar['event_type_2'].astype(np.int8)


# In[ ]:


calendar.info()


# In[ ]:


train_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
train_data.head()


# In[ ]:


def melt_item_group(group):
    numcols = [f"d_{day}" for day in range(0,1941 + 1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    
    for day in range(1942, 1942 + 28):
        group[f"d_{day}"] = np.nan

    dt = pd.melt(
        group,
        id_vars = catcols,
        value_vars = [col for col in group.columns if col.startswith("d_")],
        var_name = "d",
        value_name = "sales"
    )
    return dt


# In[ ]:


prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(np.int16)
prices['sell_price'] = prices['sell_price'].astype(np.float32)
prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(np.int16)
prices.set_index(['store_id', 'item_id', 'wm_yr_wk'], inplace=True)
prices = prices.sort_index()


# In[ ]:


prices.head()


# In[ ]:


def convert_categorical_columns(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
        df[column] = df[column].astype('category')


# In[ ]:


def get_dataset_simple(df, label_column):
    test_df = df.loc['d_1914':,:].copy()
    
    valid_df = df.loc['d_1914': 'd_1941',:].copy()
    
    train_df = df.loc[df.index[0]: 'd_1913',:].copy()
    train_df.loc[:,label_column] = train_df.loc[:,label_column]
    
    return train_df, valid_df, test_df


# In[ ]:


def get_dataset(df, label_column):
    test_df = df.loc['d_1914':,:].copy()
    
    df_shape = df.loc[:'d_1941',:].shape
    samples_idx = random.sample(range(df_shape[0]), int(0.2 * df_shape[0]))
    
    valid_idxs = samples_idx
    
    train_idxs = list(set(list(range(0, df_shape[0] ))) - set(samples_idx))
    
    return train_idxs, valid_idxs, test_df


# In[ ]:


evaluation_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

evaluation_data.loc[:,'id'] = evaluation_data['id'].apply(lambda x: x[:-10] + 'validation')
evaluation_data.set_index('id', inplace=True)
evaluation_data = evaluation_data.loc[:,'d_1914': 'd_1941'].copy()

def get_rmse(submission_validation):
    sub = submission_validation.set_index('id')
    error = mean_squared_error(sub.values, evaluation_data.loc[sub.index,:].values)
    return error


# In[ ]:


def split_prediction(prediction):
    columns = ['id'] + ['F%s' % i for i in range(1, 29)]
    prediction = prediction.reset_index().set_index('id')
    prediction_evaluation = prediction.loc[:,'d_1914':'d_1941'].reset_index().copy()
    prediction_evaluation.columns = columns
    
    prediction_validation = prediction.loc[:,'d_1942':].reset_index().copy()
    prediction_validation.columns = columns
    prediction_validation.loc[:, 'id'] = prediction_validation['id'].str[:-10] + 'validation'
    
    return prediction_validation, prediction_evaluation


# In[ ]:


def create_price_features(group_df, horizon=28):
    lag = horizon // 7
    group_df['sell_price_diff_shift_1'] = group_df.groupby('id')['sell_price'].transform(lambda x: x - x.shift(1)).astype(np.float32)
    group_df[f'sell_price_diff_shift_{horizon}'] = group_df.groupby('id')['sell_price'].transform(lambda x: x - x.shift(horizon)).astype(np.float32)
    group_df['sell_price_diff_rolling_7'] = group_df.groupby('id')['sell_price'].transform(lambda x: x - x.rolling(7).mean()).astype(np.float32)
    
    group_df[f'sell_price_diff_shift_{horizon}_shift_1'] = group_df.groupby('id')['sell_price_diff_shift_1'].transform(lambda x: x.shift(horizon)).astype(np.float32)
    group_df[f'sell_price_diff_shift_{horizon}_shift_{horizon}'] = group_df.groupby('id')[f'sell_price_diff_shift_{horizon}'].transform(lambda x: x.shift(horizon)).astype(np.float32)
    group_df[f'sell_price_diff_shift_{horizon}_rolling_7'] = group_df.groupby('id')['sell_price_diff_rolling_7'].transform(lambda x: x.shift(horizon)).astype(np.float32)
    
    group_df[f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}'] = group_df.groupby('id')['sell_price_diff_rolling_7'].transform(lambda x: x - x.shift(horizon)).astype(np.float32)


# In[ ]:


def get_created_features(horizon=28):
    feature_columns = [
        'sell_price_diff_shift_1',
        'sell_price_diff_rolling_7',
        f'sell_price_diff_shift_{horizon}',
        f'sell_price_diff_shift_{horizon}_shift_1',
        f'sell_price_diff_shift_{horizon}_shift_{horizon}',
        f'sell_price_diff_shift_{horizon}_rolling_7', # ?
        f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}',
    ]
    return feature_columns


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nidx_feature = ['id']\ncategorical_feature = [\n    'dept_id' ,\n    'state_id',\n    #'wday',\n    #'month',\n    #'year',\n    'event_name_1',\n    'event_type_1',\n    'snap_CA',\n    'snap_TX',\n    'snap_WI',\n    'event_name_2',\n    'event_type_2',\n    'is_weekend',\n    'season',\n]\nfeature_columns = categorical_feature + [\n    'sales_shift_28',\n    'sales_mean_rolling_4_wday_shift_4',\n    'sales_mean_rolling_4_wday_shift_8',\n    'sell_price',\n]\n#train_generated_features = ['day_min', 'day_mean', 'day_max']\n\nlabel_column = 'sales'\n\ndummy_subm = pd.DataFrame(columns=['id'] + ['F%s' % i for i in range(1, 29)])\nsubmission_validation = dummy_subm.copy()\nsubmission_evaluation = dummy_subm.copy()\n\nsubmission_validation_lgb = dummy_subm.copy()\nsubmission_evaluation_lgb = dummy_subm.copy()\n\ngroups = train_data.groupby('store_id')\nlen_group = len(groups)\n\nfor i, (store_id, group) in enumerate(groups):\n    print(store_id)\n\n    group = melt_item_group(group)\n    group_df = group.join(calendar.set_index('d'), on='d')\n    del group\n\n    group_df.sales.fillna(0, inplace=True)\n    \n    group_df = group_df.set_index(['d'])\n    \n    # add features for unique store\n    group_df['sales_shift_28'] = group_df.groupby(['id'])['sales'].transform(lambda x: x.shift(28)).astype(np.float32)\n    \n    group_df['sales_mean_rolling_4_wday'] = group_df.groupby(['id', 'wday'])['sales'].transform(lambda x: x.rolling(4).mean()).astype(np.float32)\n    group_df['sales_mean_rolling_4_wday_shift_4'] = group_df.groupby(['id', 'wday'])['sales_mean_rolling_4_wday'].transform(lambda x: x.shift(4)).astype(np.float32)\n    group_df['sales_mean_rolling_4_wday_shift_8'] = group_df.groupby(['id', 'wday'])['sales_mean_rolling_4_wday'].transform(lambda x: x.shift(8)).astype(np.float32)\n\n    # add prices\n    print('add prices')\n    group_df = group_df.join(\n        prices.loc[store_id], on=['item_id', 'wm_yr_wk']\n    )\n    label_cols_prices = ['state_id', 'dept_id']\n    convert_categorical_columns(group_df, label_cols_prices)\n    \n    create_price_features(group_df)\n    feature_columns += get_created_features()\n    \n    # drop rows with na\n    group_df.dropna(inplace=True)\n    print('drop rows with na')\n\n    train_idxs, valid_idxs, test_df = get_dataset(\n        group_df[feature_columns + [label_column] + idx_feature],\n        label_column=label_column\n    )\n    \n    print('get_dataset_simple')\n    #key = all_id[:-10] + 'validation'\n    prediction = test_df.reset_index().set_index(['d', 'id',])['sales_mean_rolling_4_wday_shift_4']#.values.tolist()\n    prediction = prediction.unstack(level=0)\n    prediction_validation, prediction_evaluation = split_prediction(prediction)\n    \n    try:\n        #lgb\n        dtrain = lgb.Dataset(group_df.iloc[train_idxs][feature_columns], label=group_df.iloc[train_idxs][label_column], categorical_feature=categorical_feature)\n        dvalid = lgb.Dataset(group_df.iloc[valid_idxs][feature_columns], label=group_df.iloc[valid_idxs][label_column], categorical_feature=categorical_feature)\n\n        param = {\n            'boosting_type': 'gbdt',\n            'objective': 'tweedie',\n            #'tweedie_variance_power': 1.1,\n            'metric': 'rmse',\n            'subsample': 0.5,\n            'subsample_freq': 1,\n            'learning_rate': 0.03,\n            'num_leaves': 1024,\n            'min_data_in_leaf': 1024,\n            'feature_fraction': 0.2,\n            'max_bin': 10,\n            'boost_from_average': False,\n            'verbose': -1,\n            #'lambda_l1': 0.8,\n            #'lambda_l2': 0,\n            #'min_gain_to_split': 1.,\n            #'min_sum_hessian_in_leaf': 1e-3,\n        }\n        # https://lightgbm.readthedocs.io/en/latest/index.html\n        bst = lgb.train(param, dtrain, valid_sets=[dvalid], num_boost_round = 2000, early_stopping_rounds=30, verbose_eval=True, categorical_feature=categorical_feature)\n\n        prediction_lgb = bst.predict(test_df.reset_index().set_index(['d', 'id',])[feature_columns])\n        prediction_lgb_df = test_df.reset_index()[['d', 'id']].copy()\n        prediction_lgb_df['prediction'] = prediction_lgb\n        prediction_lgb_df = prediction_lgb_df.set_index(['d', 'id',]).unstack(level=0).reset_index()\n        prediction_lgb_df.columns = ['id'] + ['d_{}'.format(i) for i in range(1914, 1970)]\n        prediction_validation_lgb, prediction_evaluation_lgb = split_prediction(prediction_lgb_df)\n    except Exception as e:\n        print(e)\n        prediction_validation_lgb = prediction_validation\n        prediction_evaluation_lgb = prediction_evaluation\n    finally:\n        submission_validation_lgb = submission_validation_lgb.append(prediction_validation_lgb)\n        submission_evaluation_lgb = submission_evaluation_lgb.append(prediction_evaluation_lgb)\n\n    submission_validation = submission_validation.append(prediction_validation)\n    submission_evaluation = submission_evaluation.append(prediction_evaluation)")


# Final:
# lgb 7.161521607829326
# rmse 4.995158482142858
# gibrid 6.87113691931274
# CPU times: user 5min 40s, sys: 53.6 s, total: 6min 33s
# Wall time: 3min 54s

# In[ ]:


submission = submission_validation.append(submission_evaluation)
submission.to_csv('/kaggle/working/submission_mean.csv', index=False)

submission_lgb = submission_validation_lgb.append(submission_evaluation_lgb)
submission_lgb.to_csv('/kaggle/working/submission_lgb.csv', index=False)


# In[ ]:




