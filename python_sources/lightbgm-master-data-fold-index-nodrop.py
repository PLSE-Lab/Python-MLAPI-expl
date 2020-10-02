#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm.notebook import tqdm_notebook as tqdm
from typing import Union
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
import feather
import random
import time
import os
import gc


# In[ ]:


#! ls /kaggle/input/m5-forecasting-accuracy


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 1414
DATA_DIR = '/kaggle/input'    
seed_everything(SEED)
FOLD = 2
EPOCHS = 200


# In[ ]:


class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            setattr(self, f'lv{i + 1}_train_df', train_df.groupby(group_id)[train_target_columns].sum())
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value'].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        train_y = getattr(self, f'lv{lv}_train_df')
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = ((train_y.iloc[:, 1:].values - train_y.iloc[:, :-1].values) ** 2).mean(axis=1)
        return (score / scale / len(valid_y)).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)


# In[ ]:


class WRMSSEForLightGBM(WRMSSEEvaluator):

    def feval(self, preds, dtrain):
        preds = preds.reshape(self.valid_df[self.valid_target_columns].shape)
        score = self.score(preds)
        return ('WRMSSE', score, False)


# In[ ]:


def get_pivot_columns(df, split):
    #dataframe.pivot()

    fix_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    cols = ['d_'+str(x) for x in np.unique(df['d'].apply(lambda x: int(x[2:])).values).tolist()]
    if split == 'valid':
        return cols
    else:
        return fix_cols+cols


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


#oof_index  = np.load(DATA_DIR+'/d-master-data-fold-index-nodrop-npy/oof_index.npy')
#test_index = np.load(DATA_DIR+'/d-master-data-fold-index-nodrop-npy/test_index.npy')


# In[ ]:


#oof = np.zeros(len(oof_index))
#prediction = np.zeros(len(test_index))
#scores = []

for i in range(FOLD):
    print("Fold", i, 'started at', time.ctime())
    
    train_idx = np.load(DATA_DIR+'/d-master-data-fold-index-nodrop-npy/train_{}.npy'.format(i))
    valid_idx = np.load(DATA_DIR+'/d-master-data-fold-index-nodrop-npy/valid_{}.npy'.format(i))
    print('Index reading done!')
    
    data = pd.read_hdf(DATA_DIR+'/d-master-data-fold-index-nodrop-h5/sales.h5', key='data')
    feats = [x for x in data.columns.values.tolist() if x not in ['date', 'part', 'id', 'target','d']]
    print('Datasets and features reading done!')

    train_x, train_y = data[feats].iloc[train_idx], data['target'].iloc[train_idx]
    valid_x, valid_y = data[feats].iloc[valid_idx], data['target'].iloc[valid_idx]
    print('Data split done!')
    
    train_cols = get_pivot_columns(data.iloc[train_idx], 'train')
    valid_cols = get_pivot_columns(data.iloc[valid_idx], 'valid')
    del data, feats, train_idx, valid_idx
    gc.collect()
    print('Pivot columns done!')

    params = {
        'learning_rate': 0.025,
        'max_depth': 10,
        'num_leaves':2**10+1,
        'metric': 'rmse',
        'random_state': SEED,
        'n_jobs':-1} 
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    
    del train_x, valid_x
    gc.collect()
    
    train_df = pd.read_csv(DATA_DIR+'/m5-forecasting-accuracy/sales_train_validation.csv')
    train_df = reduce_mem_usage(train_df, verbose=True)
    print('Raw datasets reading done!')
    train_fold_df = train_df[train_cols]
    valid_fold_df = train_df[valid_cols]
    print('Train and valid fold done!')
    del train_df
    gc.collect()
    calendar = pd.read_csv(DATA_DIR+'/m5-forecasting-accuracy/calendar.csv')
    calendar = reduce_mem_usage(calendar, verbose=True)
    prices   = pd.read_csv(DATA_DIR+'/m5-forecasting-accuracy/sell_prices.csv')
    prices   = reduce_mem_usage(prices, verbose=True)
    print('Calendar and prices reading done!')
    evaluator = WRMSSEForLightGBM(train_fold_df, valid_fold_df, calendar, prices)
    del calendar, prices
    gc.collect()
    print('Evaluator done!')
    
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=EPOCHS,
                      valid_sets = lgb_test,
                      verbose_eval=50,
                      early_stopping_rounds=50,
                      feval=evaluator.feval
                     )
    del lgb_train, lgb_test
    gc.collect()
    
    #y_pred_valid = model.predict(valid_x)
    #y_pred = model.predict(data[feats].iloc[test_index], num_iteration=model.best_iteration)

    #oof[valid_idx] = y_pred_valid.reshape(-1,)
    #scores.append(f1_score(valid_y, np.round(np.clip(y_pred_valid, 0, 10)).astype(int), average = 'macro'))
    #prediction += y_pred
    print('\n')

#prediction /= N_FOLD


# In[ ]:




