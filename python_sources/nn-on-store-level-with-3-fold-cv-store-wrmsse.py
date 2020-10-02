#!/usr/bin/env python
# coding: utf-8

# This notebook shows how to use a **neural network with categorical embeddings** and **day-to-day prediction** to forecast the sales on **store level**. To optimize the model I implemented a **3-fold cross-validation** taking into account the previous two month and the same month of the previous year as validation periods. I implemented a **store-WRMSSE** as validation metric, which is simply the error which the respective store contributes to the overall WRMSSE (regarding all data). Note that for this version the features and the model have not yet been opimized, so there is plenty of room for improvement.
# 
# The same code using a **light gradient bossting model** instead of a neural network can be found [here](https://www.kaggle.com/dantefilu/lgbm-on-store-level-with-3-fold-cv-store-wrmsse).
# 
# Here are some very useful notebooks and discussions that inspired me or where I borrowed some code. Thanks a lot!
# 
# * https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda by @headsortails
# * https://www.kaggle.com/nxrprime/preprocessing-fe by @nxrprime
# * https://www.kaggle.com/ragnar123/very-fst-model by @ragnar123
# * https://www.kaggle.com/kneroma/m5-forecast-v2-python by @kneroma
# * https://www.kaggle.com/tnmasui/m5-wrmsse-evaluation-dashboard by @tnmasui
# * https://www.kaggle.com/mayer79/m5-forecast-keras-with-categorical-embeddings-v2 by @mayer79
# * https://www.kaggle.com/nxrprime/transformer-xl-intro-baseline by @nxrprime
# 
# * https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/141515 by @amedprof
# * https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/142129 by @kneroma
# * https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/143077 by @hengck23
# * https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/138881 by @kyakovlev
# * https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/144067 by @kyakovlev
# 

# <a id="TOC"></a>
# ## Table of contents
# * **1. [Set global parameters](#global)** <br>
# * **2. [Import libraries and data](#Import)** <br>
# * **3. [Initialize WRMSSE](#WRMSSE)** <br>
# * **4. [Data analysis and feature engineering](#Ana)** <br>
#     * 4.1 [Sales](#Ana1) <br>
#     * 4.2 [Calendar](#Ana2) <br>
#     * 4.3 [Prices](#Ana3) <br>
# * **5. [Data preparation for modeling](#Prep)** <br>
#     * 5.1 [Merging Sales, Calendar, and Prices data sets](#Prep1) <br>
#     * 5.2 [Encoding](#Prep2) <br>   
#     * 5.3 [Perpare data for CV and final submission](#Prep3) <br>
# * **6. [Modeling](#Model)** <br>
#     * 5.1 [The Model](#Model1) <br>
#     * 5.2 [Cross-validation: training and prediction](#Model2) <br>
# * **7. [Conclusions](#Conclusions)** <br>
# 
# 

# ## 1. Set global parameters <a id="global"></a>
# 
# 

# In[ ]:


# target store 
STORE = 'CA_1'

# first and last days which are considered for training data
FIRST_TRAIN_DAY=600
LAST_TRAIN_DAY= 1941

# set cross validation conditions
VALID_SET_LENGTH = 28
CV_STEPS = 3

# path for data import
PATH_TO_FILE = "../input/m5-forecasting-accuracy"


# [back to Table of Contents](#TOC)
# ## 2. Import libraries and data <a id="Import"></a>

# In[ ]:


import pandas as pd
import numpy as np
import gc
import os
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model


# In[ ]:


# credit to https://www.kaggle.com/ragnar123/very-fst-model
def reduce_memory_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df.loc[:,col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df.loc[:,col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df.loc[:,col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df.loc[:,col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df.loc[:,col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df.loc[:,col] = df[col].astype(np.float32)
                else:
                    df.loc[:,col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df  


# In[ ]:


def load_sales_data(path=PATH_TO_FILE):
    print('Load sales data ...')
    sales_all = pd.read_csv(os.path.join(path, "sales_train_evaluation.csv"))
    sales=sales_all[sales_all.store_id==STORE].copy()
    return sales, sales_all

def load_calendar_data(path=PATH_TO_FILE):
    print('Load calendar data ...')
    calendar_all = pd.read_csv(os.path.join(path, "calendar.csv"))
    if 'CA' in STORE:
        calendar=calendar_all.drop(['snap_TX','snap_WI'], axis=1)    
    elif 'TX' in STORE:
        calendar=calendar_all.drop(['snap_CA','snap_WI'], axis=1) 
    elif 'WI' in STORE:
        calendar=calendar_all.drop(['snap_CA','snap_TX'], axis=1)
    return calendar, calendar_all


def load_price_data(path=PATH_TO_FILE):
    print('Load price data ...')
    prices_all = pd.read_csv(os.path.join(path, "sell_prices.csv"))
    prices=prices_all[prices_all.store_id==STORE]
    return prices, prices_all


# In[ ]:


get_ipython().run_cell_magic('time', '', '# load all data\nsales, sales_all = load_sales_data()\ncalendar, calendar_all = load_calendar_data()\nprices, prices_all = load_price_data()\n\n# data needed for evaluation process\nsample_submission = pd.read_csv(os.path.join(PATH_TO_FILE, "sample_submission.csv"))\nsales_copy = sales.copy()')


# [back to Table of Contents](#TOC)
# ## 3. Calculate weights for WRMSSE <a id="WRMSSE"></a>

# Credit mainly goes to https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and https://www.kaggle.com/tnmasui/m5-wrmsse-evaluation-dashboard

# In[ ]:


class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 'all'  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')]                     .columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')]                               .columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], 
                                 axis=1, sort=False)

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
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)                    [valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns]                    .set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index()                   .rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left',
                                    on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd'])                    .unstack(level=2)['value']                    .loc[zip(self.train_df.item_id, self.train_df.store_id), :]                    .reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns],
                               weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt) 
    
    def score(self, valid_preds: Union[pd.DataFrame, 
                                       np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape                == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, 
                                       columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], 
                                 valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):

            valid_preds_grp = valid_preds.groupby(group_id)[self.valid_target_columns].sum()
            setattr(self, f'lv{i + 1}_valid_preds', valid_preds_grp)
            
            lv_scores = self.rmsse(valid_preds_grp, i + 1)
            setattr(self, f'lv{i + 1}_scores', lv_scores)
            
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, 
                                  sort=False).prod(axis=1)
            
            all_scores.append(lv_scores.sum())
            
        self.all_scores = all_scores

        return np.mean(all_scores)


# In[ ]:


def prepare_store_prediction_for_WRMSSE_dashboard(pred, sales_all=sales_all): 
    # prepare template with ground truth according to validation step
    sub = sales_all[['id']+[f'd_{d}' for d in range(pred.d.min(), pred.d.max()+1)]]
    sub = sub[sub.id.str.endswith('_evaluation')].copy()
    sub = sub.set_index(['id'])

    # transfom prediction df for score evaluation
    pred = pred.assign(id=pred.id, d_="d_" + (pred.d).astype("str"))
    pred = pred.pivot(index='id', columns="d_", values="sales")

    # update respective store with predicted values
    sub.update(pred)

    # reindex
    new_index=sales_all[['id','item_id','store_id']].copy()
    sub = sub.reset_index().merge(new_index, how='left', on=['id']).set_index(['item_id','store_id'])
    sub = sub.drop(['id'], axis=1)

    return sub


# In[ ]:


get_ipython().run_cell_magic('time', '', "WRMSSE_evaluators = {}\nfor CV_step in range(1,CV_STEPS+1):\n    print(f'Build WRMSSE_evaluator for CV step {CV_step} ...')\n    \n    # define last CV step: same 28 days as for prediction period but of the previous year\n    CV_start = FIRST_TRAIN_DAY\n    if CV_step == CV_STEPS:\n        CV_end = LAST_TRAIN_DAY - 364 + VALID_SET_LENGTH   \n    else:\n        CV_end = LAST_TRAIN_DAY - (CV_step-1)*VALID_SET_LENGTH \n        \n    print(f'validation start day: {CV_end-VALID_SET_LENGTH+1}')\n    print(f'validation end day: {CV_end}')\n    \n    # prepare train and valid data for current CV step\n    train_df = sales_all.drop([f'd_{i}' for i in range(1,CV_start)]\n                              +[f'd_{i}' for i in range(CV_end - VALID_SET_LENGTH +1,LAST_TRAIN_DAY+1)], axis=1)\n    #train_df = train_df.drop([f'd_{i}' for i in range(CV_train_end,LAST_TRAIN_DAY)], axis=1)\n    valid_df = sales_all[[f'd_{i}' for i in range(CV_end - VALID_SET_LENGTH +1,CV_end +1)]]\n    \n    # prepare calendar for current CV step\n    calendar_n = calendar_all.iloc[ CV_start-1 : CV_end, : ]\n\n    # prepare prices for current CV step\n    prices_start = calendar_all['wm_yr_wk'][CV_start-1]\n    prices_end = calendar_all['wm_yr_wk'][CV_end-1]\n    prices_n = prices_all.loc[(prices_all.wm_yr_wk>=prices_start) & (prices_all.wm_yr_wk<=prices_end), :]\n    \n    # inizialize WRMSSE_evaluator objects for current CV step\n    WRMSSE_evaluators[f'eval_CV_{CV_step}'] = WRMSSEEvaluator(train_df, valid_df, calendar_n, prices_n)")


# In[ ]:


# visualization of WRMSSE

def create_viz_df(df,lv):
    
    df = df.T.reset_index()
    if lv in [6,7,8,9,11,12]:
        df.columns = [i[0] + '_' + i[1] if i != ('index','')                       else i[0] for i in df.columns]
    df = df.merge(calendar_all.loc[:, ['d','date']], how='left', 
                  left_on='index', right_on='d')
    df['date'] = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df.drop(['index', 'd'], axis=1)
    
    return df

def create_dashboard(CV_step):
    
    evaluator = WRMSSE_evaluators[f'eval_CV_{CV_step}']
    wrmsses = [np.mean(evaluator.all_scores)] + evaluator.all_scores
    labels = ['Overall'] + [f'Level {i}' for i in range(1, 13)]

    
        
    # configuration array for the charts
    n_rows = [1, 1, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3]
    n_cols = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    width = [7, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
    height = [4, 3, 12, 3, 9, 9, 9, 9, 9, 9, 9, 9]
    
    for i in [3,8]:
        
        scores = getattr(evaluator, f'lv{i}_scores')
        weights = getattr(evaluator, f'lv{i}_weight')
        
        if i > 1 and i < 9:
            if i < 7:
                fig, axs = plt.subplots(1, 2, figsize=(12, 3))
            else:
                fig, axs = plt.subplots(2, 1, figsize=(12, 8))
                
            ## RMSSE plot
            scores.plot.bar(width=.8, ax=axs[0], color='g')
            axs[0].set_title(f"RMSSE", size=14)
            axs[0].set(xlabel='', ylabel='RMSSE')
            if i >= 4:
                axs[0].tick_params(labelsize=8)
            for index, val in enumerate(scores):
                axs[0].text(index*1, val+.01, round(val,4), color='black', 
                            ha="center", fontsize=10 if i == 2 else 8)
            
            ## Weight plot
            weights.plot.bar(width=.8, ax=axs[1])
            axs[1].set_title(f"Weight", size=14)
            axs[1].set(xlabel='', ylabel='Weight')
            if i >= 4:
                axs[1].tick_params(labelsize=8)
            for index, val in enumerate(weights):
                axs[1].text(index*1, val+.01, round(val,2), color='black', 
                            ha="center", fontsize=10 if i == 2 else 8)
                    
            fig.suptitle(f'Level {i}: {evaluator.group_ids[i-1]}', size=24 ,
                         y=1.1, fontweight='bold')
            plt.tight_layout()
            plt.show()

    # PLOT SALES
    # actual data
    sales_cum= data[(data.d<=data_val[f'CV_{CV_step}'].d.max())&(data.d>=data_val[f'CV_{CV_step}'].d.max()-60)].groupby('d')['sales'].sum()

    # NN prediction 
    sales_cum_NN_pred = data_val[f'CV_{CV_step}'].groupby('d')['sales'].sum()

    ##seasonal naive baseline
    #shift=364
    #sales_cum_baseline = data[(data.d<=data_val[f'CV_{CV_step}'].d.max()-shift)&(data.d>data_val[f'CV_{CV_step}'].d.max()-shift-VALID_SET_LENGTH)].groupby('d')['sales'].sum()
    #sales_cum_baseline.index = sales_cum_baseline.index +shift

    polt = sales_cum.plot(x ='d', y='sales', kind = 'line',figsize=(16,8), label='actual sales')
    polt = sales_cum_NN_pred.plot(x ='d', y='sales', kind = 'line',figsize=(16,8), label='predicted sales: NN')
    #polt = sales_cum_baseline.plot(x ='d', y='sales', kind = 'line',figsize=(16,8), label='predicted sales: seasonal naive')
    plt.legend(loc="upper left")
    plt.title(f'All sales for store {STORE}')
    plt.show()


# [back to Table of Contents](#TOC)
# ## 4. Data analysis and feature engineering <a id="Ana"></a>

# ### 4.1 Sales data <a id="Ana1"></a>

# In[ ]:


sales.head()


# In[ ]:


sales.info()


# In[ ]:


def prepare_sales_data(df, first_train_day=FIRST_TRAIN_DAY):
    
    # select rows for selected store
    df = df.drop(['state_id', 'store_id'], axis=1)
    
    # add day columns for test data - this way both train and test data are prepared together
    last_train_day = int(df.columns[-1].replace('d_',''))
    for day in range(last_train_day+1, last_train_day+ 28 +1):
        df[f"d_{day}"] = np.nan
    
    # reshape sales days 'd_XXXX' from wide to long format.
    if first_train_day != None:
        df = df.drop(["d_" + str(i + 1) for i in range(first_train_day)], axis=1).copy()
    id_vars = [col for col in df.columns if not col.startswith("d_")]
    df = df.melt(id_vars = id_vars, var_name = "d", value_name = "sales")
    
    # convert feature 'd' to integer
    df = df.assign(d=df.d.str[2:].astype("int16"))
    
    # reduce data size
    df = reduce_memory_usage(df)
    
    return df


# In[ ]:


sales = prepare_sales_data(sales)
gc.collect()
sales.head()


# In[ ]:



def create_sales_features(df, params):
    if params['activate']:
        lag = params['lag']
        print(f'shift_lag: {lag}')
        if params['lag_feature'] == True:
            df.loc[:,f'lag_l{lag}'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(lag))
        if params['roll_mean_win'] != []:
            for win in params['roll_mean_win']:
                print(f'rolling_mean: {lag}/{win}')                                                                       
                df[f'rolling_mean_l{lag}_w{win}'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(lag).rolling(win).mean())
        if params['roll_median_win'] != []:
            for win in params['roll_median_win']:
                print(f'rolling_median: {lag}/{win}')                                                                       
                df[f'rolling_median_l{lag}_w{win}'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(lag).rolling(win).median())
        if params['roll_std_win'] != []:
            for win in params['roll_std_win']:
                print(f'rolling_std: {lag}/{win}')                                                                       
                df[f'rolling_std_l{lag}_w{win}'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(lag).rolling(win).std())
    return df   


# In the following cell you can set the parameters for the sales features: 
# Select **'activate': True** to use the respective lag feature and fill the list with whatever windows you want to use. If the list is empty, it will not be used.
# The code below allows to easily try a lot of different lag features and always be sure that the missing values are calculated in the day-to-day prediction for lags smaller than 28 days (more infos on this issue [here](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/141515))

# In[ ]:


# set parameters for sales features and activate for use

params_lag7 = {'activate': True,
               'lag': 7,
               'lag_feature':True,
               'roll_mean_win':[7],
               'roll_median_win':[7],
               'roll_std_win':[7]}

params_lag14 = {'activate': False,
               'lag': 14,
               'lag_feature':True,
               'roll_mean_win':[7],
               'roll_median_win':[7],
               'roll_std_win':[7]}

params_lag21 = {'activate': False,
               'lag': 21,
               'lag_feature':True,
               'roll_mean_win':[7],
               'roll_median_win':[7],
               'roll_std_win':[7]}

params_lag28 = {'activate': True,
               'lag': 28,
               'lag_feature':True,
               'roll_mean_win':[7,28],
               'roll_median_win':[7,28],
               'roll_std_win':[7,28]}

params_lag182 = {'activate': False,
               'lag': 182,
               'lag_feature':True,
               'roll_mean_win':[28],
               'roll_median_win':[],
               'roll_std_win':[]}

params_lag364 = {'activate': False,
               'lag': 364,
               'lag_feature':True,
               'roll_mean_win':[7,28],
               'roll_median_win':[],
               'roll_std_win':[]}


# create features
sales = create_sales_features(sales, params_lag7)
sales = create_sales_features(sales, params_lag14)
sales = create_sales_features(sales, params_lag21)
sales = create_sales_features(sales, params_lag28)
sales = create_sales_features(sales, params_lag182)
sales = create_sales_features(sales, params_lag364)


# In[ ]:


def return_df_with_lag_and_win_per_column(df):
    df_shift = pd.DataFrame({'lag+win': [0 for col in df.columns]}, index = df.columns)
    for col in df.columns:
        if '_l' in col:
            pre = col.split('_l')[1]
            if '_w' in pre:
                lag = int(pre.split('_w')[0])
                win = int(pre.split('_w')[1])             
                total = lag + win
            else: 
                lag = int(pre)
                win = 0
                total = lag
            df_shift.loc[col, 'lag'] = lag
            df_shift.loc[col, 'win'] = win
            df_shift.loc[col, 'lag+win'] = total
    return df_shift.loc[df_shift['lag+win']!=0]
        


# In[ ]:


# remove rows where NaNs originate from feature engineering
mask = (return_df_with_lag_and_win_per_column(sales)['lag+win']).idxmax()
sales = sales[(sales.d >= 1914) | (pd.notna(sales[mask]))].copy()


# In[ ]:


sales.info()


# ### 4.2 Calendar data <a id="Ana2"></a>

# In[ ]:


calendar.head()


# In[ ]:


calendar.info()


# In[ ]:


def prepare_calendar_data(df):
    #drop further useless features  
    df = df.drop(['date', 'weekday'], axis=1)
    
    # convert feature 'd' to integer
    df = df.assign(d = df.d.str[2:].astype(int))
    
    # substitute NaN in event_name and event_type by 'undefined'
    df = df.fillna('undefined')
    
    # reduce data size
    df = reduce_memory_usage(df)
    
    return df


# In[ ]:


calendar = prepare_calendar_data(calendar)
gc.collect()
calendar.head()


# In[ ]:


calendar.info()


# ### 4.3 Price data <a id="Ana3"></a>

# In[ ]:


prices.head()


# In[ ]:


prices.info()


# In[ ]:


def prepare_price_data(df):
    #drop useless features  
    df = df.drop(['store_id'], axis=1)
    
    # reduce data size
    df = reduce_memory_usage(df)
    
    return df


# In[ ]:


prices = prepare_price_data(prices)


# In[ ]:


prices.sell_price.isnull().value_counts()


# In[ ]:


def create_price_features(df):
    
    # percentage change between the current and a prior element
    df["sell_price_rel_diff"] = df.groupby(["item_id"])["sell_price"].pct_change()
    
    # rolling std of prices
    df["sell_price_roll_sd7"] = df.groupby(["item_id"])["sell_price"].transform(lambda x: x.rolling(7).std())
    
    # relative cumulative price 
    grouped = df.groupby(["item_id"])["sell_price"]
    df["sell_price_cumrel"] = (grouped.shift(0) - grouped.cummin()) / (1 + grouped.cummax() - grouped.cummin())
    
    # reduce data size
    df = reduce_memory_usage(df)
    
    return df


# In[ ]:


#create new price features
prices = create_price_features(prices)


# In[ ]:


prices.info()


# [back to Table of Contents](#TOC)
# ## 5. Data preparation for modeling <a id="Prep"></a>

# ### 5.1 Merge Sales, calendar and prices data sets <a id='Prep1'></a>

# In[ ]:


data = sales.merge(calendar, how="left", on="d")
data = data.merge(prices, how="left", on=["wm_yr_wk", "item_id"])
data.drop(["wm_yr_wk"], axis=1, inplace=True)
del sales, calendar, prices
gc.collect()
data.head()


# ### 5.2 Encoding <a id='Prep2'></a>

# In[ ]:


cat_features = ["item_id", "dept_id", "cat_id", 
                'wday', 'month', 'year', 
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
bool_features = [f"snap_{STORE.split('_')[0]}"]
num_features = [e for e in data.columns if e not in cat_features + bool_features + ['d','sales','id']]
dense_features = num_features + bool_features


# encode numerical features 
for i, feature in enumerate(num_features):
    data[feature] = data[feature].fillna(data[feature].median())
    #data[feature] = StandardScaler().fit_transform(data[[feature]])
    

# encode categorical features
for i, feature in enumerate(cat_features):
    data[feature] = OrdinalEncoder(dtype="int").fit_transform(data[[feature]])


data = reduce_memory_usage(data)


# ### 5.3 Prepare data for cross-validation and final submission <a id='Prep3'></a>

# In[ ]:


def check_number_of_rows_with_NaNs(df, column='', last_train_day=LAST_TRAIN_DAY):
    if column == '':
        NaN_rows_all = len(df[(df.d<=last_train_day) & df.isna().any(axis=1)])
        print('rows containing NaN values: '+ str(NaN_rows_all))
    else:
        NaN_rows_all = len(df[(df.d<=last_train_day) & df.isna().any(axis=1)])
        NaN_rows_col = len(df[(df.d<=last_train_day) & df[column].isna()])
        print('rows containing NaN values (column/total): '+ str(NaN_rows_col)+'/'+ str(NaN_rows_all))
        
check_number_of_rows_with_NaNs(data)


# In[ ]:


# further functions for data preparation which will be called individualy 
# before the respective training/prediction run (see below)

def create_input_dictionary_for_model(df):
    X = {"dense_features": df[dense_features].to_numpy()}
    for i, feature in enumerate(cat_features):
        X[feature] = df[[feature]].to_numpy()
    return X

def prepare_data_for_current_validation_step(data, current_CV_step):
    # determine column with max number of NaN caused by lag features
    df_lag_win = return_df_with_lag_and_win_per_column(data)
    max_NaN_rows = max(df_lag_win['lag+win'])
        
    # determin start and end days of training and validation data for current validation step
    if current_CV_step == CV_STEPS: 
        first_day = LAST_TRAIN_DAY - max_NaN_rows - 364 -1
        last_day = LAST_TRAIN_DAY + VALID_SET_LENGTH - 364
        pred_start_day =  last_day - VALID_SET_LENGTH +1
    else:
        first_day = LAST_TRAIN_DAY - current_CV_step*VALID_SET_LENGTH - max_NaN_rows -1
        last_day = LAST_TRAIN_DAY - (current_CV_step-1)*VALID_SET_LENGTH
        pred_start_day =  last_day - VALID_SET_LENGTH +1
   
    print(f'Validation range: {pred_start_day} - {last_day}\n')
    print('Prepare data ...\n')

    # prepare training data
    mask = data.d < pred_start_day
    X_train = create_input_dictionary_for_model(data[mask])
    y_train = data["sales"][mask]
    
    # prepare validation data for model
    mask = (data.d >= pred_start_day) & (data.d <= last_day) 
    X_valid = create_input_dictionary_for_model(data[mask])
    y_valid = data["sales"][mask]

   # prepare validation data for realistic day-to-day prediction (to avoid data leakage)  
    df_val = data[(data.d >= first_day) & (data.d <= last_day)].copy()
    df_val.loc[df_val.d >= pred_start_day, 'sales']=0
    for col, lag in df_lag_win.lag.items():   
        df_val.loc[df_val.d > pred_start_day + lag, col]=0

    return X_train, y_train, X_valid, y_valid, df_val


# In[ ]:


gc.collect()


# [back to Table of Contents](#TOC)
# ## 6. Modeling <a id='Model'></a>

# ### 6.1 The Model <a id='Model1'></a>

# In[ ]:


def create_model(loss=keras.losses.mean_squared_error,
                 metrics=['mse'],
                 optimizer=keras.optimizers.Nadam(learning_rate=0.0002)
                ):
    
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    gc.collect()

    # Numerical features 
    dense_input = Input(shape=(len(dense_features), ), name='dense_features')

    # Categorical embeddings
    item_id_input = Input(shape=(1,), name='item_id')
    item_id_emb = Flatten()(Embedding(len(data['item_id'].unique()), 3)(item_id_input))

    dept_id_input = Input(shape=(1,), name='dept_id')
    dept_id_emb = Flatten()(Embedding(len(data['dept_id'].unique()), 1)(dept_id_input))

    cat_id_input = Input(shape=(1,), name='cat_id')
    cat_id_emb = Flatten()(Embedding(len(data['cat_id'].unique()), 1)(cat_id_input))

    wday_input = Input(shape=(1,), name='wday')
    wday_emb = Flatten()(Embedding(len(data['wday'].unique()), 1)(wday_input))

    month_input = Input(shape=(1,), name='month')
    month_emb = Flatten()(Embedding(len(data['month'].unique()), 1)(month_input))

    year_input = Input(shape=(1,), name='year')
    year_emb = Flatten()(Embedding(len(data['year'].unique()), 1)(year_input))

    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_name_1_emb = Flatten()(Embedding(len(data['event_name_1'].unique()), 1)(event_name_1_input))

    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_type_1_emb = Flatten()(Embedding(len(data['event_type_1'].unique()), 1)(event_type_1_input))

    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_name_2_emb = Flatten()(Embedding(len(data['event_name_2'].unique()), 1)(event_name_2_input))

    event_type_2_input = Input(shape=(1,), name='event_type_2')
    event_type_2_emb = Flatten()(Embedding(len(data['event_type_2'].unique()), 1)(event_type_2_input))

    inputs = [dense_input, wday_input, month_input,year_input, event_name_1_input, event_type_1_input,
              event_name_2_input, event_type_2_input, item_id_input, dept_id_input, cat_id_input]
    
    # Combine numerical inputs and embedding 
    x = concatenate([dense_input, wday_emb, month_emb, year_emb, event_name_1_emb, event_type_1_emb, 
                     event_name_2_emb, event_type_2_emb, item_id_emb, dept_id_emb,cat_id_emb])

    # Define dense layers 
    x = BatchNormalization()(x)
    x = Dense(30, activation="elu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(30, activation="elu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(30, activation="elu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(10, activation="elu", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation="linear", name='output')(x)

    # Connect input and output
    model = Model(inputs, output)
    
    # compile model
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    
    return model


# ### 6.2 Cross-validatoion: Training and prediction <a id='Model2'></a>

# In[ ]:



def predict_validation_data_for_current_validation_step(data, current_CV_step):
    
    # prepare data
    X_train, y_train, X_valid, y_valid, data_val = prepare_data_for_current_validation_step(data, current_CV_step)

    
    # compile model
    print('Compile model ...\n')
    model = create_model(loss=keras.losses.mean_squared_error,
                         metrics=['mse'],
                         optimizer=keras.optimizers.Nadam(learning_rate=0.0002))

    
    # train model
    print('Train model ...\n')
    history = model.fit(X_train, 
                        y_train,
                        batch_size=10000,
                        epochs=20,
                        shuffle=True,
                        validation_data=(X_valid,y_valid))


    # day-by-day prediction
    print('\n Predict validation data ...\n')
    lag = 7 # min lag used for features
    interval = 1

    if current_CV_step == CV_STEPS: 
        start_day = LAST_TRAIN_DAY -364 +1
    else:
        start_day = LAST_TRAIN_DAY - current_CV_step*VALID_SET_LENGTH +1

    for i in range(start_day, data_val.d.max() +1):
        print(i)
        if i== start_day + interval*lag:
            data_val = create_sales_features(data_val, params_lag7)
            interval+=1  
        if i== start_day + 2*lag:
            data_val = create_sales_features(data_val, params_lag14)
        if i== start_day + 3*lag:
            data_val = create_sales_features(data_val, params_lag21)

        X = create_input_dictionary_for_model(data_val[data_val.d == i])
        y_pred = model.predict(X, batch_size=10000)

        data_val.loc[data_val.d == i, "sales"] = y_pred.clip(0)
    
    print('\n')

    # calculate store-WRMSSE for CV step
    pred = prepare_store_prediction_for_WRMSSE_dashboard(data_val[data_val.d>=start_day]).values
    W = WRMSSE_evaluators[f'eval_CV_{current_CV_step}'].score(pred) 
    print(f'store-WRMSSE for CV_{current_CV_step}: ' + str(W)+'\n\n\n')
    
    # summarize history for accuracy
    plt.plot(history.history['mse'], color='red')
    plt.plot(history.history['val_mse'], color='green')
    plt.plot(history.history['loss'], color='red')
    plt.plot(history.history['val_loss'], color='green')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train accuracy/loss', 'validation accuracy/loss',], loc='best')
    plt.show()
       
    return data_val[data_val.d>=start_day], [W]


# In[ ]:


get_ipython().run_cell_magic('time', '', "data_val={}\nCV_scores=pd.DataFrame()\n\nfor CV_step in range(1,CV_STEPS+1):\n    print(f'VALIDATION STEP {CV_step}\\n')\n    \n    # training and prediction for CV step\n    data_val[f'CV_{CV_step}'], CV_scores[f'{STORE}_CV{CV_step}'] = predict_validation_data_for_current_validation_step(data,CV_step)")


# In[ ]:


# show store-WRMSSE of the different CV steps (store-WRMSSE is the contribution of the respective store to the total/final WRMSSE)
CV_scores[f'{STORE}_CV_mean'] = CV_scores.loc[0,:].mean()
CV_scores.to_csv(f"wrmsse_{STORE}.csv", index=True)
CV_scores


# In[ ]:


# select CV step that you want to display
CV_step = 2
create_dashboard(CV_step)


# [back to Table of Contents](#TOC)
# ## 7. Conclusions <a id="Conclusions"></a>

# In order to predict the whole data set (all stores), you can optimize the model for each store in an individual notebook and combine the predictions in one submission file (not implemented here). Making predictions on store level and running the different notebooks in parallel allows a quick optimization process and solves the problem with RAM limitation here on kaggle. Note that for this version neither the features nor the model have been opimized. For further ideas for FE see the references below in the comments.
# 

# In[ ]:





# In[ ]:




