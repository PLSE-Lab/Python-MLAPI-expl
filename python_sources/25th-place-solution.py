#!/usr/bin/env python
# coding: utf-8

# ### Library

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import datetime, random, math
from catboost import CatBoostClassifier
import lightgbm as lgb
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import gc, pickle
import ast
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from sklearn.linear_model import Ridge,Lasso, BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


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
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
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
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores


# ### utils

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

def set_index(df, name):
    d = {}
    for col, value in df.iloc[0,:].items():
        try:
            if '_evaluation' in value:
                d[col] = 'id'
            if 'd_' in value:
                d[col] = 'd'
        except:
            if type(value)!=str:
                d[col]=name
    return d

def dcol2int(col):
    if col[:2]=='d_':
        return int(col.replace('d_', ''))
    else:
        return col
    
def str_category_2_int(data):
    categories = [c for c in data.columns if data[c].dtype==object]
    for c in categories:
        if c=='id' or c=='d':
            pass
        else:
            data[c] = pd.factorize(data[c])[0]
            data[c] = data[c].replace(-1, np.nan)
    return data

def select_near_event(x, event_name):
    z = ''
    for y in x:
        if y in event_name:
            z+=y+'_'
    if len(z)==0:
        return np.nan
    else:
        return z
    
def sort_d_cols(d_cols):
    d_cols = [int(d.replace('d_','')) for d in d_cols]
    d_cols = sorted(d_cols)
    d_cols = [f'd_{d}' for d in d_cols]
    return d_cols


# ### Preprocessing

# In[ ]:


def preprocessing(path, d_cols):
    train_d_cols = d_cols[-(200+63):]
    train_df = pd.read_csv(path+'sales_train_evaluation.csv')
    calendar_df = pd.read_csv(path+'calendar.csv')
    sell_prices_df = pd.read_csv(path+'sell_prices.csv')
    sell_prices_df['price'] = sell_prices_df['sell_price']
    del sell_prices_df['sell_price']
    sample_submission_df = pd.read_csv(path+'sample_submission.csv')
    
    train_df.index = train_df.id
    calendar_df['date']=pd.to_datetime(calendar_df.date)
    calendar_df.index = calendar_df.d
    
    
    cat_cols = [ col for col in train_df.columns if 'id' in str(col)]
    new_columns = cat_cols+d_cols
    train_df = train_df.reindex(columns=new_columns)
    
    data = train_df[train_d_cols].stack(dropna=False).reset_index()
    data = data.rename(columns=set_index(data, 'TARGET'))
    data.reset_index(drop=True, inplace=True)
    
    data['wm_yr_wk'] = data.d.map(calendar_df.set_index('d')['wm_yr_wk'])
    for key, value in train_df[['item_id','dept_id', 'cat_id', 'state_id', 'store_id']].to_dict().items():
        data[key] = data.id.map(value)
    data = pd.merge(data, sell_prices_df[['store_id', 'item_id', 'wm_yr_wk', 'price']], on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    data.drop('wm_yr_wk', axis=1, inplace=True)
    data['dept_id_price'] = data['price']/data.groupby(['d', 'dept_id', 'store_id'])['price'].transform('mean')
    data['cat_id_price'] = data['price']/data.groupby(['d', 'cat_id', 'store_id'])['price'].transform('mean')
    data['is_sell_cnt'] = data.groupby(['dept_id', 'store_id', 'd'])['price'].transform(lambda x: x.notnull().sum())
    
    #snap_data
    snap_data = calendar_df[['snap_CA', 'snap_WI', 'snap_TX', 'd']]
    snap_data.set_index('d', inplace=True)
    data[f'snap']=0
    for key, value in snap_data.to_dict().items():
        k = key.replace('snap_', '')
        data.loc[data.state_id==k,'snap'] = data.loc[data.state_id==k, 'd'].map(value).fillna(0)
    
    
    event_name = ['SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart', 'LentWeek2', 'StPatricksDay', 'Purim End', 
              'OrthodoxEaster', 'Pesach End', 'Cinco De Mayo', "Mother's day", 'MemorialDay', 'NBAFinalsStart', 'NBAFinalsEnd',
              "Father's day", 'IndependenceDay', 'Ramadan starts', 'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'Halloween', 
              'EidAlAdha', 'VeteransDay', 'Thanksgiving', 'Christmas', 'Chanukah End', 'NewYear', 'OrthodoxChristmas', 
              'MartinLutherKingDay', 'Easter']
    event_type = ['Sporting', 'Cultural', 'National', 'Religious']
    event_names = {'event_name_1':event_name, 'event_type_1':event_type}
    for event, event_name in event_names.items():
        for w in [4]:
            calendar_df[f'new_{event}_{w}']=''
            for i in range(-1,-(w+1),-1):
                calendar_df[f'new_{event}_{w}'] += calendar_df[event].shift(i).astype(str)+'|'
            calendar_df[f'new_{event}_{w}'] = calendar_df[f'new_{event}_{w}'].apply(lambda x: x.split('|'))
            calendar_df[f'new_{event}_{w}'] = calendar_df[f'new_{event}_{w}'].apply(lambda x: select_near_event(x, event_name))

    #calendar_dict
    cols = ['new_event_name_1_4', 'new_event_type_1_4', 'wday', 'month', 'year', 'event_name_1','event_type_1']
    for key, value in calendar_df[cols].to_dict().items():
        data[key] = data.d.map(value)
    for shift in [-1,1]:
        data[f'snap_{shift}'] = data.groupby(['id'])['snap'].shift(shift)
        data[f'snap_{shift}'] = data[f'snap_{shift}'].fillna(0)
    
    data = reduce_mem_usage(data)
    gc.collect()
    return data


# ### FE

# In[ ]:


def make_roll_data(data, win):
    data_2 = data.groupby(['id'])['TARGET'].apply(
            lambda x:
            x.shift(1).rolling(win, min_periods=1).agg({'mean'})
        )
    for col in data_2.columns:
        data[f'roll_{win}_{col}'] = data_2[col]
        
    return data

def shift_diff_data(data):
    data[f'shift_diff']=0
    for i in range(4):
        data[f'shift_diff'] += data.groupby(['id'])['TARGET'].apply( lambda x :x.diff(7*i).shift(7) )/4
    return data
    

def make_lag_roll_data(data, lag):
    data[f'lag{lag}_roll_14_mean'] = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        x.shift(lag).rolling(28, min_periods=1).mean()
    )
    
    return data

def make_shift_data(data):
    for i in range(0,10,2):
        data[f'shift_{7*(i+1)}'] = data.groupby(['id'])['TARGET'].shift(7*(i+1))
        
    return data


def fe(data):
    data = make_roll_data(data, 7)
    data = make_roll_data(data, 28)
    data = shift_diff_data(data)
    data = make_lag_roll_data(data, 56)
    data = make_lag_roll_data(data, 84)
    data = make_shift_data(data)
    
    return data


# ### Train utils

# In[ ]:


def plot_importance(models, col):
    importances = np.zeros(len(col))
    for model in models:
        importances+=model.feature_importance(importance_type='gain')
    importance = pd.DataFrame()
    importance['col'] = col
    importance['importance'] = minmax_scale(importances)
    #importance.to_csv(f'importance_{name}.csv',index=False)
    return importance

def predict_cv(x_val, models):
    preds = np.zeros(len(x_val))
    for model in models:
        pred = model.predict(x_val)
        preds+=pred/len(models)
    return preds

def train_predict_RE(data, PARAMS):
    days = sorted(data.d.unique())
    days = sort_d_cols(days)
    trn_days = days[:-28]
    val_days = days[-28:]
        
    for i in range(28):
        data = fe(data)
        if i==0:
            shift_cols = [col for col in data.columns if 'shift' in col]
            roll_cols = [col for col in data.columns if 'roll' in col]
            features=['dept_id', 'store_id', 'snap', 'snap_1', 'dept_id_price', 'cat_id_price', 'price',
                      'new_event_type_1_4','event_name_1','event_type_1', 'wday', 
                      'is_sell_cnt'
                     ]+shift_cols+roll_cols
            models=[]
            print(f' FEATIRE LEN {len(features)}')
            X = data[data.d.isin(trn_days)][data.TARGET.notnull()]
            X.reset_index(drop=True, inplace=True)
            train_set = lgb.Dataset(X.loc[:,features], X.loc[:,'TARGET'])
            for cy in range(3):
                PARAMS['random_state']=2020+cy
                model = lgb.train(train_set=train_set, params=PARAMS,verbose_eval=500)
                models.append(model)
        
        val_day = val_days[i]
        predict_data = data[data.d==val_day]
        preds = predict_cv(predict_data[features], models)
        data.loc[data.d==val_day, 'TARGET'] = preds
        
    sub = data[data.d.isin(val_days)][['id', 'd', 'TARGET', 'price']]
    return sub


# ### Preprocessing  Group

# In[ ]:


def create_is_sell_data(sell_prices_df, calendar_df, train_df):
    train_df.index = train_df['id']
    sell_prices_df['id'] = sell_prices_df['item_id'].astype('str')+'_'+sell_prices_df['store_id']+'_evaluation'
    sell_prices_data = sell_prices_df[sell_prices_df.wm_yr_wk.isin(calendar_df.wm_yr_wk.unique())]
    sell_prices_data.reset_index(drop=True, inplace=True)
    tmp = sell_prices_data.groupby(['id'])[['wm_yr_wk', 'sell_price']].apply(
        lambda x: x.set_index('wm_yr_wk')['sell_price'].to_dict()
    ).to_dict()
    d = calendar_df.d
    wm_yr_wk = calendar_df.wm_yr_wk
    price_data = {}
    for col in tqdm(train_df.id.unique()):
        price_data[col] = wm_yr_wk.map(tmp[col])
    price_data = pd.DataFrame(price_data)
    price_data.index = d
    is_sell = price_data.notnull().astype(float).T
    price_data = price_data.fillna(0)
    
    is_sell.index=train_df.id
    train_df.index=train_df.id
    is_sell = pd.concat([
        train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']], is_sell
    ], axis=1)
    price_data = pd.concat([
        train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']], price_data.T  
    ], axis=1)
    
    return price_data, is_sell
    
def calendar_preprocessing(calendar):
    calendar['qaurter'] = pd.to_datetime(calendar['date']).dt.day.apply(lambda x: x//7)

    event_name = ['SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart', 'LentWeek2', 'StPatricksDay', 'Purim End', 
                'OrthodoxEaster', 'Pesach End', 'Cinco De Mayo', "Mother's day", 'MemorialDay', 'NBAFinalsStart', 'NBAFinalsEnd',
                "Father's day", 'IndependenceDay', 'Ramadan starts', 'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'Halloween', 
                'EidAlAdha', 'VeteransDay', 'Thanksgiving', 'Christmas', 'Chanukah End', 'NewYear', 'OrthodoxChristmas', 
                'MartinLutherKingDay', 'Easter']
    event_type = ['Sporting', 'Cultural', 'National', 'Religious']
    event_names = {'event_name_1':event_name, 'event_type_1':event_type}
    for event, event_name in event_names.items():
        calendar[f'new_{event}']=''
        for i in range(-1,-5,-1):
            calendar[f'new_{event}'] += calendar[event].shift(i).astype(str)+'|'
        calendar[f'new_{event}'] = calendar[f'new_{event}'].apply(lambda x: x.split('|'))
        calendar[f'new_{event}'] = calendar[f'new_{event}'].apply(lambda x: select_near_event(x, event_name))
    return calendar

class M5_Data:
    def __init__(self, path, d_cols):
        train = pd.read_csv(path+'sales_train_evaluation.csv')
        calendar = pd.read_csv(path+'calendar.csv')
        price = pd.read_csv(path+'sell_prices.csv')
        self.price_data, self.is_sell = create_is_sell_data(price, calendar, train)

        self.d_cols = d_cols
        train = train.reindex(
            columns=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']+self.d_cols
        )

        train = train.set_index('id', drop=False)
        self.train = pd.concat([
            train[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']],
            train[self.d_cols]*self.price_data[self.d_cols]
        ], axis=1)
        self.calendar = calendar_preprocessing(calendar)
        
        self.make_all_id()
        
    def make_all_id(self):
        self.train['all_id'] = 'all_id'
        self.price_data['all_id'] = 'all_id'
        
    
    def make_id(self, id_1, id_2):
        new_id = id_1+'X'+id_2
        self.train[new_id] = self.train[id_1].astype(str)+'X'+self.train[id_2].astype(str)
        self.price_data[new_id] = self.price_data[id_1].astype(str)+'X'+self.price_data[id_2].astype(str)


# ### FE Group

# In[ ]:


def sale_cnt_by_group(ID, data, m5_data, trn_d_cols, group='state_id'):
    for _id in m5_data.train[group].unique():
        f = m5_data.train[m5_data.train[group]==_id].groupby(ID)[trn_d_cols].sum(min_count=1).T
        f = f.stack(dropna=False).reset_index().rename(columns={0:f'TARGET_{_id}', 'level_0':'d'})
        data = pd.merge(data, f, on=['d', ID])
        
        f = m5_data.price_data[m5_data.price_data[group]==_id].replace(0, np.nan).groupby(ID)[trn_d_cols].count().T
        f = f.stack(dropna=False).reset_index().rename(columns={0:f'cnt_{_id}', 'level_0':'d'})
        data = pd.merge(data, f, on=['d', ID])
    return data

def fe_group(ID, data, log=False):
    target_cols = [col for col in data.columns if 'TARGET' in col]
    for target_col in target_cols:
        if log:
            data[target_col] = np.log1p(data[target_col])
        for win in [7,28]:
            agg = {'mean'}
            data_2 = data.groupby(ID)[target_col].apply(
                lambda x: x.rolling(win, min_periods=1).agg(agg)
            )
            for col in data_2.columns:
                data[f'roll{win}_{col}_{target_col}'] = data_2[col]
        data[f'roll28_mean_{target_col}_lag56'] = data.groupby(ID)[f'roll28_mean_{target_col}'].shift(56)
        data[f'roll28_mean_{target_col}_lag84'] = data.groupby(ID)[f'roll28_mean_{target_col}'].shift(84)
        
        for i in range(0,10,2):
            data[f'shift{7*(i+1)}_{target_col}'] = data.groupby(ID)[target_col].shift(7*i)
                
        data[f'shift_diff_{target_col}']=0
        for i in range(4):
            data[f'shift_diff_{target_col}'] += data.groupby(ID)[target_col].diff(7*i)/4
    
    del_f = [col for col in target_cols if col!='TARGET']
    data.drop(columns=del_f, inplace=True)
    
    return data

def map_calendar_data(ID, data, m5_data, map_1, map_2):
    """
    map_1=['qaurter','wday','month','year','new_event_name_1', 
    'new_event_type_1','event_name_1','event_type_1',
    'snap_CA','snap_WI','snap_TX']
    map_2=['event_name_1','event_type_1','snap_CA','snap_WI','snap_TX']
    """
    for key, value in m5_data.calendar.set_index('d')[map_1].items():
        data[key] = data.d.map(value)
        if data[key].dtypes==object:
            data[key] = pd.factorize(data[key])[0]
        if key in map_2:
            for i in [-1,1]:
                data[f'{key}_{i}'] = data.groupby(ID)[key].shift(i)
    return data

def make_snap_data(ID, data, m5_data):
    data['snap'] = 0
    for key, value in m5_data.calendar.set_index('d')[['snap_CA','snap_WI','snap_TX']].items():
        state = key.replace('snap_', '')
        data.loc[data[ID].str.contains(state),'snap'] = data.loc[data[ID].str.contains(state),'d'].map(value)
    for i in [ -1,1]:
        data[f'snap_{i}'] = data.groupby(ID)['snap'].shift(i)
    return data

def make_data_dept_cat(ID, trn_d_cols, m5_data):
    data = m5_data.train.groupby(ID)[trn_d_cols].sum(min_count=1).T
    data = data.stack(dropna=False).reset_index().rename(columns={0:'TARGET', 'level_0':'d'})
    f = m5_data.price_data.replace(0,np.nan).groupby(ID)[
        trn_d_cols
    ].count().stack(dropna=False).reset_index().rename(columns={0:'cnt', 'level_1':'d', 'level_0':'d'})
    data = pd.merge(data, f, on=['d', ID])
    
    data = sale_cnt_by_group(ID, data, m5_data, trn_d_cols)
    data = fe_group(ID, data, log=True)

    map_1 = ['qaurter','wday','month','year','new_event_name_1',
             'new_event_type_1','event_name_1','event_type_1',
             'snap_CA','snap_WI','snap_TX']
    map_2 = ['event_name_1','event_type_1','snap_CA','snap_WI','snap_TX']
    data = map_calendar_data(ID, data, m5_data, map_1, map_2)
    
    data[f'f_{ID}'] = pd.factorize(data[ID])[0]
    data = data[data.d.isin(trn_d_cols[28:])]
    return data    

def make_data_all_id(ID, trn_d_cols, m5_data):
    data = m5_data.train.groupby(ID)[trn_d_cols].sum(min_count=1).T
    data = data.stack(dropna=False).reset_index().rename(columns={0:'TARGET', 'level_0':'d'})
    f = m5_data.price_data.replace(0,np.nan).groupby(ID)[
        trn_d_cols
    ].count().stack(dropna=False).reset_index().rename(columns={0:'cnt', 'level_1':'d', 'level_0':'d'})
    data = pd.merge(data, f, on=['d', ID])
    
    data = sale_cnt_by_group(ID, data, m5_data, trn_d_cols)
    data = fe_group(ID, data, log=False)
    
    map_1 = ['qaurter','wday','month','year','new_event_name_1',
             'new_event_type_1','event_name_1','event_type_1',
             'snap_CA','snap_WI','snap_TX']
    map_2 = ['event_name_1','event_type_1','snap_CA','snap_WI','snap_TX']
    data = map_calendar_data(ID, data, m5_data, map_1, map_2)

    data[f'f_{ID}'] = pd.factorize(data[ID])[0]
    data = data[data.d.isin(trn_d_cols[28:])]
    return data

def make_data_state_store(ID, trn_d_cols, m5_data):
    
    data = m5_data.train.groupby(ID)[trn_d_cols].sum(min_count=1).T
    data = data.stack(dropna=False).reset_index().rename(columns={0:'TARGET', 'level_0':'d'})
    f = m5_data.price_data.replace(0,np.nan).groupby(ID)[
        trn_d_cols
    ].count().stack(dropna=False).reset_index().rename(columns={0:'cnt', 'level_1':'d', 'level_0':'d'})
    data = pd.merge(data, f, on=['d', ID])
    
    group='cat_id'
    for _id in m5_data.train[group].unique():
        f = m5_data.train[m5_data.train[group]==_id].groupby(ID)[trn_d_cols].sum(min_count=1).T
        f = f.stack(dropna=False).reset_index().rename(columns={0:f'TARGET_{_id}', 'level_0':'d'})
        data = pd.merge(data, f, on=['d', ID])
    
    for _id in m5_data.price_data[group].unique():
        f = m5_data.price_data[m5_data.price_data[group]==_id].replace(0, np.nan).groupby(ID)[trn_d_cols].count().T
        f = f.stack(dropna=False).reset_index().rename(columns={0:f'cnt_{_id}', 'level_0':'d'})
        data = pd.merge(data, f, on=['d', ID])
        
    data = fe_group(ID, data, log=False)
    map_1 = ['qaurter','wday','month','year','new_event_name_1', 'new_event_type_1','event_name_1','event_type_1']
    map_2 = ['event_name_1','event_type_1']
    data = map_calendar_data(ID, data, m5_data, map_1, map_2)
    data = make_snap_data(ID,data, m5_data)
    
    data[f'f_{ID}'] = pd.factorize(data[ID])[0]
    data = data[data.d.isin(trn_d_cols[28:])]
    return data

def make_data_2_id(ID, trn_d_cols, m5_data):
    
    data = m5_data.train.groupby(ID)[trn_d_cols].sum(min_count=1).T
    data = data.stack(dropna=False).reset_index().rename(columns={0:'TARGET', 'level_0':'d'})
    f = m5_data.price_data.replace(0,np.nan).groupby(ID)[
        trn_d_cols
    ].count().stack(dropna=False).reset_index().rename(columns={0:'cnt', 'level_1':'d', 'level_0':'d'})
    data = pd.merge(data, f, on=['d', ID])
    
    data = fe_group(ID, data, log=True)
    map_1 = ['qaurter','wday','month','year','new_event_name_1', 'new_event_type_1','event_name_1','event_type_1']
    map_2 = ['event_name_1','event_type_1']
    data = map_calendar_data(ID, data, m5_data, map_1, map_2)
    data = make_snap_data(ID, data, m5_data)
    
    data[f'f_{ID}'] = pd.factorize(data[ID])[0]
    data = data[data.d.isin(trn_d_cols[28:])]
    return data

def make_data_item(ID, trn_d_cols, m5_data):
    
    data = m5_data.train.groupby(ID)[trn_d_cols].sum(min_count=1).T
    data = data.stack(dropna=False).reset_index().rename(columns={0:'TARGET', 'level_0':'d'})
    f = m5_data.price_data.replace(0,np.nan).groupby(ID)[
        trn_d_cols
    ].count().stack(dropna=False).reset_index().rename(columns={0:'cnt', 'level_1':'d', 'level_0':'d'})
    data = pd.merge(data, f, on=['d', ID])
    
    data = fe_group(ID, data, log=False)
    map_1 = ['qaurter','wday','new_event_name_1',
             'new_event_type_1','event_name_1','event_type_1',
             'snap_CA','snap_WI','snap_TX']
    map_2 = ['event_name_1','event_type_1','snap_CA','snap_WI','snap_TX']
    data = map_calendar_data(ID, data, m5_data, map_1, map_2)
    
    data[f'f_{ID}'] = pd.factorize(data[ID])[0]
    data = data[data.d.isin(trn_d_cols[28:])]
    return data


# ### flow for lavels

# In[ ]:


def all_flow_dept_cat(m5_data, trn_d_cols, PARAMS):
    Results={}
    for ID in ['dept_id', 'cat_id']:
        data = make_data_dept_cat(ID, trn_d_cols, m5_data)
        sub =train_predict_group(data, ID, params=PARAMS[ID], n_split=5, log=True)
        Results[ID] = sub
    return Results

def all_flow_all_id(m5_data, trn_d_cols, PARAMS):
    ID='all_id'
    Results={}
    data = make_data_all_id(ID, trn_d_cols, m5_data)
    sub =train_predict_all(data, ID, params=PARAMS['all_id'], n_split=5, log=False)
    Results['all_id'] = sub
    return Results

def all_flow_2_id(m5_data, trn_d_cols, PARAMS):
    Results = {}
    for id_1 in ['dept_id', 'cat_id']:
        for id_2 in ['store_id','state_id']:
            m5_data.make_id(id_1, id_2)
            ID = id_1+'X'+id_2
            data = make_data_2_id(ID, trn_d_cols, m5_data)
            sub =train_predict_group(data, ID, PARAMS[ID], n_split=5, log=True)
            Results[ID] = sub
    return Results

def all_flow_store_state(m5_data, trn_d_cols, PARAMS):
    Results ={}
    for ID in ['store_id', 'state_id']:
        data = make_data_state_store(ID, trn_d_cols, m5_data)
        sub =train_predict_group(data, ID, PARAMS[ID], n_split=5, log=False)
        Results[ID] = sub
    return Results

def all_flow_item(m5_data, trn_d_cols, PARAMS):
    Results = {}
    ID = 'item_id'
    data = make_data_item(ID, trn_d_cols, m5_data)
    sub =train_predict_item(data, ID, PARAMS[ID], n_split=5, log=False)
    Results[ID] = sub
    return Results


# ### Group train

# In[ ]:


def predict_cv_group(x_val, models, log):
    preds = np.zeros(len(x_val))
    for model in models:
        pred = model.predict(x_val)
        if log:
            preds+=(np.e**(pred)-1)/len(models)
        else:
            preds+=pred/len(models)
            
    return preds

def train_predict_all(data,ID,params,n_split=5,log=False, cy=2):
    days = data.d.unique().tolist()
    days = sort_d_cols(days)
    trn_days = days[:-28]
    val_days = days[-28:]
    data.reset_index(drop=True, inplace=True)

    shift_cols = [col for col in data.columns if 'shift' in col]
    roll_cols = [col for col in data.columns if 'roll' in col]
    cat_cols = [col for col in data.columns if (not 'shift' in col) and (not 'roll' in col)]
    cat_cols = [col for col in cat_cols if not col in [ID, 'd', 'TARGET']]
    features=cat_cols+shift_cols+roll_cols
    
    for i in range(28):
        if i%7==0:
            data[shift_cols] = data.groupby(ID)[shift_cols].shift(7)
        data[roll_cols] = data.groupby(ID)[roll_cols].shift(1)
            
        models=[]
        X = data[data.d.isin(trn_days)][data.TARGET.notnull()]
        
        split = X[X.d.isin(trn_days[-500:])]['TARGET']
        split = split.mean()-3.*split.std()
        X = X[X.TARGET>split]
        X.reset_index(drop=True, inplace=True)
        groups =X['wday'].astype(str)
        y = (500*minmax_scale(X['TARGET'])).astype(int)
        k = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2020)
        for _cy_ in range(cy):
            params['random_state'] = 2020+_cy_
            for trn, val in k.split(X,y=y, groups=groups):
                train_set = lgb.Dataset(X.loc[:,features], X.loc[:,'TARGET'])
                val_set = lgb.Dataset(X.loc[val,features], X.loc[val,'TARGET'])

                model = lgb.train(train_set=train_set, valid_sets=[train_set, val_set], params=params, num_boost_round=3000, 
                                  early_stopping_rounds=100, verbose_eval=500)
                models.append(model)
                
        val_day = val_days[i]
        predict_data = data[data.d==val_day]
        preds = predict_cv_group(predict_data[features], models, log)
        
        data.loc[data.d==val_day, 'TARGET'] = preds
        
    sub = data[data.d.isin(val_days)][[ID, 'd', 'TARGET']]
    sub = sub.groupby(ID)[['d', 'TARGET']].apply(lambda x: x.set_index('d')['TARGET'].T)[val_days]
    sub = sub.reset_index()
    return sub

def train_predict_item(data,ID,params,n_split=5,log=False):
    days = data.d.unique().tolist()
    days = sort_d_cols(days)
    trn_days = days[:-28]
    val_days = days[-28:]
    data.reset_index(drop=True, inplace=True)

    shift_cols = [col for col in data.columns if 'shift' in col]
    roll_cols = [col for col in data.columns if 'roll' in col]
    cat_cols = [col for col in data.columns if (not 'shift' in col) and (not 'roll' in col)]
    cat_cols = [col for col in cat_cols if not col in [ID, 'd', 'TARGET']]
    features=cat_cols+shift_cols+roll_cols
    
    for i in range(28):
        if i%7==0:
            data[shift_cols] = data.groupby(ID)[shift_cols].shift(7)
        data[roll_cols] = data.groupby(ID)[roll_cols].shift(1)
            
        models=[]
        X = data[data.d.isin(trn_days)][data.TARGET.notnull()]
        
        split = X[X.d.isin(trn_days[-500:])]['TARGET']
        split = split.mean()-3.*split.std()
        X = X[X.TARGET>split]
        X.reset_index(drop=True, inplace=True)
        groups =X['wday'].astype(str)
        y = (500*minmax_scale(X['TARGET'])).astype(int)
        k = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2020)
        c=0
        params['random_state'] = 2020
        for trn, val in k.split(X,y=y, groups=groups):
            train_set = lgb.Dataset(X.loc[trn,features], X.loc[trn,'TARGET'])
            val_set = lgb.Dataset(X.loc[val,features], X.loc[val,'TARGET'])
            model = lgb.train(train_set=train_set, valid_sets=[train_set, val_set], params=params, num_boost_round=3000, 
                              early_stopping_rounds=100, verbose_eval=500)
            models.append(model)
            c+=1
            if c==3:
                break
                
        val_day = val_days[i]
        predict_data = data[data.d==val_day]
        preds = predict_cv_group(predict_data[features], models, log)
        
        data.loc[data.d==val_day, 'TARGET'] = preds
        
    sub = data[data.d.isin(val_days)][[ID, 'd', 'TARGET']]
    sub = sub.groupby(ID)[['d', 'TARGET']].apply(lambda x: x.set_index('d')['TARGET'].T)[val_days]
    sub = sub.reset_index()
    return sub

def train_predict_group(data,ID,params,n_split=5,log=False, cy=3):
    days = data.d.unique().tolist()
    days = sort_d_cols(days)
    trn_days = days[:-28]
    val_days = days[-28:]
    data.reset_index(drop=True, inplace=True)
    
    shift_cols = [col for col in data.columns if 'shift' in col]
    roll_cols = [col for col in data.columns if 'roll' in col]
    cat_cols = [col for col in data.columns if (not 'shift' in col) and (not 'roll' in col)]
    cat_cols = [col for col in cat_cols if not col in [ID, 'd', 'TARGET']]
    features=cat_cols+shift_cols+roll_cols
    
    for i in range(28):
        if i%7==0:
            data[shift_cols] = data.groupby(ID)[shift_cols].shift(7)
        data[roll_cols] = data.groupby(ID)[roll_cols].shift(1)
            
        models=[]
        X = data[data.d.isin(trn_days)][data.TARGET.notnull()]
        
        split = X[X.d.isin(trn_days[-500:])]['TARGET']
        split = split.mean()-3.*split.std()
        X = X[X.TARGET>split]
        X.reset_index(drop=True, inplace=True)
        groups =X['wday'].astype(str)
        y = (500*minmax_scale(X['TARGET'])).astype(int)
        k = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2020)
        for _cy_ in range(cy):
            params['random_state'] = 2020+_cy_
            for trn, val in k.split(X,y=y, groups=groups):
                train_set = lgb.Dataset(X.loc[trn,features], X.loc[trn,'TARGET'])
                val_set = lgb.Dataset(X.loc[val,features], X.loc[val,'TARGET'])

                model = lgb.train(train_set=train_set, valid_sets=[train_set, val_set], params=params, num_boost_round=3000, 
                                  early_stopping_rounds=100, verbose_eval=500)
                models.append(model)
                
        val_day = val_days[i]
        predict_data = data[data.d==val_day]
        preds = predict_cv_group(predict_data[features], models, log)
        
        data.loc[data.d==val_day, 'TARGET'] = preds
        
    sub = data[data.d.isin(val_days)][[ID, 'd', 'TARGET']]
    sub = sub.groupby(ID)[['d', 'TARGET']].apply(lambda x: x.set_index('d')['TARGET'].T)[val_days]
    sub = sub.reset_index()
    return sub


# ### PARAMS

# In[ ]:


PARAMS = {'boosting_type': 'gbdt', 
          'objective' : 'poisson',
          #'objective' : 'tweedie', 'tweedie_variance_power': 1.141893486974509,
          'random_state':2020,
          "metric" :"rmse", "force_row_wise" : True, "learning_rate" : 0.075, "sub_row" : 0.75, "bagging_freq" : 1,
          "lambda_l2" : 0.1, 'verbosity': 1, 'num_iterations' : 2500}

# optimized using optuna
PARAMS_GROUP={}
PARAMS_GROUP['dept_id'] =  {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse',
                'max_bin': 100,'n_estimators': 1400,'boost_from_average': False,'verbose': -1,'random_state':2020,
                'subsample': 0.8897026631967412, 'subsample_freq': 0.42708068942389565,
                'learning_rate': 0.030223062885783494,'num_leaves': 66, 'feature_fraction': 0.4294129948533598,
                'bagging_freq': 4, 'min_child_samples': 11, 'lambda_l2': 6.563593012634628}
PARAMS_GROUP['cat_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse',
              'max_bin': 100,'n_estimators': 1400,'boost_from_average': False,'verbose': -1,'random_state':2020,
              'subsample': 0.8915155260035615, 'subsample_freq': 0.7106654494817621, 'learning_rate': 0.0439216532905,
              'num_leaves': 24, 'feature_fraction': 0.48092257402284877, 'bagging_freq': 7, 'min_child_samples': 6,
              'lambda_l2': 0.009774011952810685}
PARAMS_GROUP['all_id'] =  {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                     'boost_from_average': False,'verbose': -1,'random_state':2020,
                     'subsample': 0.7936430506570977, 'subsample_freq': 0.5206476443073623, 
                     'learning_rate': 0.09653801757509976,'num_leaves': 8, 'feature_fraction': 0.4986072155764939,
                     'bagging_freq': 3, 'min_child_samples': 8,'lambda_l2': 1.319732403794593}
PARAMS_GROUP['store_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                      'boost_from_average': False,'verbose': -1,'random_state':2020,
                      'subsample': 0.7936430506570977, 'subsample_freq': 0.5206476443073623, 'learning_rate': 0.09653801757509976,
                      'num_leaves': 8, 'feature_fraction': 0.4986072155764939, 'bagging_freq': 3, 'min_child_samples': 8,
                      'lambda_l2': 1.319732403794593}
PARAMS_GROUP['state_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                      'boost_from_average': False,'verbose': -1,'random_state':2020,
                      'subsample': 0.7836421955591786, 'subsample_freq': 0.7516149374096728, 'learning_rate': 0.030576348046803408,
                      'num_leaves': 11, 'feature_fraction': 0.4114638046420348, 'bagging_freq': 2, 'min_child_samples': 7, 
                      'lambda_l2': 0.13216719821842418}
PARAMS_GROUP['cat_idXstore_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                        'boost_from_average': False,'verbose': -1,'random_state':2020,
                        'subsample': 0.8976728421649643, 'subsample_freq': 0.7133363789924351, 'learning_rate': 0.066033852,
                        'num_leaves': 30, 'feature_fraction': 0.401368826100617, 'bagging_freq': 7, 'min_child_samples': 12,
                        'lambda_l2': 0.0002899138139549539}
PARAMS_GROUP['cat_idXstate_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                        'boost_from_average': False,'verbose': -1,'random_state':2020,
                        'subsample': 0.7304828007643646, 'subsample_freq': 0.7194392041924155, 'learning_rate': 0.065824,
                        'num_leaves': 20, 'feature_fraction': 0.9953802109485864, 'bagging_freq': 1, 'min_child_samples': 6,
                        'lambda_l2': 0.00026247413038444007}
PARAMS_GROUP['dept_idXstore_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                                  'boost_from_average': False,'verbose': -1,'random_state':2020,
                                  'subsample': 0.767911185577838, 'subsample_freq': 0.7287605276865973, 'learning_rate': 0.041153778,
                                  'num_leaves': 82,'feature_fraction': 0.5709048571103937, 'bagging_freq': 1, 'min_child_samples': 17,
                                  'lambda_l2': 0.45656192806764545}
PARAMS_GROUP['dept_idXstate_id'] = {'boosting_type': 'gbdt','objective': 'rmse','metric': 'rmse','max_bin': 100,'n_estimators': 1400,
                                  'boost_from_average': False,'verbose': -1,'random_state':2020,
                                  'subsample': 0.8901855563568182, 'subsample_freq': 0.5413800115150902, 'learning_rate': 0.0814993056342757,
                                  'num_leaves': 32, 'feature_fraction': 0.5087751807175098, 'bagging_freq': 7, 'min_child_samples': 70,
                                  'lambda_l2': 0.014539412980849776}

PARAMS_GROUP['item_id'] = {'boosting_type': 'gbdt', 'objective': 'tweedie', 'metric': 'rmse', 'max_bin': 100, 
                     'n_estimators': 2000, 'boost_from_average': False, 'verbose': -1, 'random_state': 2020,
                     'tweedie_variance_power': 1.141893486974509, 'subsample': 0.8710431222390667, 
                     'subsample_freq': 0.5692738176797527, 'learning_rate': 0.10957379305366494, 'num_leaves': 8,  
                     'feature_fraction': 0.45380044045308154, 'bagging_freq': 4, 'min_child_samples': 5, 
                     'lambda_l1': 7.510525772813387e-06, 'lambda_l2': 4.1004528526443944e-07}


# In[ ]:


def all_flow_all_preds(path, d_cols):
    
    data = preprocessing(path, d_cols)
    data = str_category_2_int(data)
    
    use_days=data.d.unique().tolist()
    use_days=sort_d_cols(use_days)[63:]
    data = data[data.d.isin(use_days)]
    gc.collect()
    
    mem = data.memory_usage().sum()/1024**2
    print(f"""
    DATA SHAPE   {data.shape}
    MEMORY USAGE   {mem:.2f}MB
    DATA COLUMNS  {data.columns.tolist()}
    """)
    
    gc.collect()
    all_preds = train_predict_RE(data=data, PARAMS=PARAMS)
    return all_preds

def all_flow_group(path, d_cols, PARAMS_GROUP):
    m5_data = M5_Data(path, d_cols)
    Sub={}
    
    trn_d_cols = d_cols[-200:]
    rslt = all_flow_item(m5_data, trn_d_cols, PARAMS_GROUP)
    Sub.update(rslt)
    
    trn_d_cols = d_cols[-730:]
    rslt = all_flow_2_id(m5_data, trn_d_cols, PARAMS_GROUP)
    Sub.update(rslt)
    rslt = all_flow_dept_cat(m5_data, trn_d_cols, PARAMS_GROUP)
    Sub.update(rslt)
    rslt = all_flow_store_state(m5_data, trn_d_cols, PARAMS_GROUP)
    Sub.update(rslt)
    
    
    trn_d_cols = d_cols[-900:]
    rslt = all_flow_all_id(m5_data, trn_d_cols, PARAMS_GROUP)
    Sub.update(rslt)
    
    return Sub

def main():
    
    path = '../input/m5-forecasting-accuracy/'

    d_cols=[f'd_{i+1}' for i in range(1969)]
    
    
    group_preds = all_flow_group(path, d_cols, PARAMS_GROUP)
    with open(f'group_preds.pickle', 'wb') as f:
        pickle.dump(group_preds, f)
    gc.collect()
    
    all_preds = all_flow_all_preds(path, d_cols)
    gc.collect()
    
    cat=pd.read_csv(path+'sales_train_evaluation.csv').set_index('id')[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    for key, value in cat.items():
        all_preds[key] = all_preds['id'].map(value)
    all_preds['all_id'] = 'all_id'
    all_preds.to_csv('all_preds.csv', index=False)
    
    all_preds['TARGET_original'] = all_preds['TARGET']*all_preds['price']
    all_preds['TARGET'] = all_preds['TARGET'].apply(lambda x: 0 if x<0.05 else x)
    all_preds['TARGET']*=all_preds['price']
    
    for ID in ['dept_id', 'cat_id', 'store_id', 'state_id', 'all_id', 'item_id']:
        data = group_preds[ID]
        data = data.set_index(ID)
        data = data.stack().reset_index().rename(columns={'level_1':'d', 0:f'{ID}_preds'})
        all_preds = pd.merge(all_preds,data,on=['d', ID] )
        if ID=='item_id':
            all_preds[f'{ID}_preds'] = all_preds[f'{ID}_preds']*(all_preds['TARGET_original']/all_preds.groupby(['d', ID])['TARGET_original'].transform('sum'))
        else:
            all_preds[f'{ID}_preds'] = all_preds[f'{ID}_preds']*(all_preds['TARGET']/all_preds.groupby(['d', ID])['TARGET'].transform('sum'))

    for id_1 in ['dept_id', 'cat_id']:
        for id_2 in ['store_id', 'state_id']:
            ID = f'{id_1}X{id_2}'
            all_preds[ID] = all_preds[id_1].astype(str)+'X'+all_preds[id_2].astype(str)
            data = group_preds[ID]
            data = data.set_index(ID)
            data = data.stack().reset_index().rename(columns={'level_1':'d', 0:f'{ID}_preds'})
            all_preds = pd.merge(all_preds,data,on=['d', ID] )
            all_preds[f'{ID}_preds'] = all_preds[f'{ID}_preds']*(all_preds['TARGET']/all_preds.groupby(['d', ID])['TARGET'].transform('sum'))

    p_1 = 0.35/8
    p_2 = 0.65/2
    all_preds['TARGET_2'] = p_2*all_preds['dept_idXstore_id_preds']+p_2*all_preds['cat_idXstore_id_preds']+                      p_1*all_preds['dept_id_preds']+p_1*all_preds['dept_idXstate_id_preds']+                      p_1*all_preds['store_id_preds']+p_1*all_preds['store_id_preds']+                      p_1*all_preds['cat_id_preds']+p_1*all_preds['all_id_preds']+p_1*all_preds['TARGET']+p_1*all_preds['item_id_preds']
   
    d_cols=sort_d_cols(all_preds.d.unique())
    all_preds['TARGET_2'] = all_preds['TARGET_2']/all_preds['price']
    sub = all_preds.groupby(['id'])[['d', 'TARGET_2']].apply(lambda x: x.set_index('d')['TARGET_2'].T)
    sub.columns=d_cols
    
    sample_sub=pd.read_csv(path+'sample_submission.csv')
    sample_sub = sample_sub.set_index('id', drop=False)
    sample_sub.loc[sub.index,[f'F{i}' for i in range(1,29)]] = sub[d_cols].values
    sample_sub.reset_index(drop=True, inplace=True)
    
    sample_sub.to_csv('submission.csv', index=False)


# In[ ]:


main()

