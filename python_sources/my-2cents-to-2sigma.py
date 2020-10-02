#!/usr/bin/env python
# coding: utf-8

# This notebook presents a LGBM model where both the `market` and `news` data are merged. In order to facilitate the data preparation and the subsequent update during prediction stage, the model is embedded in a python Class. In the first part of the notebook, the Class funcionalities are presented, step-by-step, by adding new methods to the class instance. In a second stage, the real training is performed.
# 
# <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Utility functions](#1)
# 1. [Class definition](#2)
#     1. [Selecting training dates](#3)
#     1. [Feature engineering](#4)
#         1. [Engineering the News data](#5)
#         1. [merging the News and Market data](#6)
#         1. [ Indexing categorical columns](#7)
#         1. [ Lagged information](#8)
#     1. [Feature selection](#9)
# 1. [Training the model](#10)
# 1. [Model pipeline](#11)
# 1. [Test set predictions](#12)

# In[ ]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from joblib import parallel, delayed
import gc
import sys
import pytz
import warnings
import time
import inspect
import datetime
from itertools import chain
from datetime import date, timedelta
from kaggle.competitions import twosigmanews
warnings.filterwarnings("ignore",category=DeprecationWarning)
#______________________________________________________________
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()


# <a id="1"></a> <br>
# ## 1. Utility functions
# We first load some utility functions. The first, written by [Guillaume Martin](https://www.kaggle.com/gemartin/load-data-reduce-memory-usage#), allows to manage memory:

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


# The second helps tracking memory usage:

# In[ ]:


def show_mem_usage():
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    list_objects=[]
    mem = 0
    for x in globals():
        if x.startswith('_'): continue
        if x in sys.modules: continue 
        if x in ipython_vars: continue
        if isinstance(globals().get(x), pd.DataFrame):
            mem = sys.getsizeof(globals().get(x))/1e+6
            if mem > 1:
                list_objects.append([x, mem ])
        else:
            for o in dir(globals().get(x)):
                if o.startswith('__'): continue
                mem = sys.getsizeof(getattr(globals().get(x), o))/1e+6
                if mem > 1:
                    list_objects.append(['.'.join([x,o]), mem ])
    return sorted(list_objects, key=lambda x: x[1], reverse=True)


# Finally, we introduce a decorator that allow to time functions:

# In[ ]:


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if TIME_FUNCTIONS:
            dt = (te - ts)*1000
            if dt < 1000:
                print ('{:<40}  {:>20.2f} ms'.format(method.__name__, dt))
            else:
                print ('{:<40}  {:>20.2f} s'.format(method.__name__, dt/1000))
        return result
    return timed


# We can check in which objects the memory is currently allocated: 

# In[ ]:


gc.collect()
memory_used = show_mem_usage()
print("approximate memory usage: {:>5.2f}GB".format(sum([s[1] for s in memory_used])/1000))
memory_used


# <a id="2"></a> <br>
# ## 1. Class definition
# In this section, I develop a class that contains the feature engineering and training of the model. For a didactic purpose, the various methods are are introduced consecutively and are thus added to the instance that contains the model. In production, all the methods would be merged while first defining the class.

# In[ ]:


class model2SigmaStockPrizes():
    #_______________________
    # class initialisation
    @timeit
    def __init__(self, market, news, verbose=False):
        self.market = market.sort_values('time')
        self.news = news.sort_values('time')
        self.market = reduce_mem_usage(self.market, verbose)
        self.news = reduce_mem_usage(self.news, verbose)
        self.verbose = verbose
        self._format_dates()
        self._convert_booleans()
    #________________________________________________________________________
    # detection of columns with dates and times, then reduced to single dates
    @timeit
    def _format_dates(self):
        for df in [self.market, self.news]:
            datetime_cols = [c for c in df.columns if 'date' in str(df[c].dtypes)]
            for col in datetime_cols:
                df[col] = df[col].dt.normalize()
                if self.verbose: 
                    print ("Content of column:'{}' set as date".format(col))
    #_________________________________
    # convert booleans columns to int
    @timeit
    def _convert_booleans(self):
        for col in self.news.columns:
            if self.news[col].dtype == bool: 
                self.news[col] = self.news[col].astype(int)


# We create the class instance:

# In[ ]:


TIME_FUNCTIONS = True
two_sigma_model =  model2SigmaStockPrizes(market_train, news_train, verbose=True)


# In[ ]:


del news_train
del market_train
gc.collect()
show_mem_usage()


# <a id="3"></a> <br>
# ### 1.1 Selecting training dates
# To reduce the time spent to pre-process the data, and ease the FE exploration, we can limit the number of dates during the exploration. Moreover, as outlined in a few discussions and kernels, the data before 2009 may false the models and it is safe to forget about this period. Hence, the following method allows to select the first date to account for:

# In[ ]:


@timeit
def select_dates(self, first_date):
    self.market = self.market.loc[self.market['time'] >= first_date]
    self.news = self.news.loc[self.news['time'] >= first_date]
    if self.verbose:
        print("data before '{}' has been removed".format(first_date))
#______________________________________________________________________
two_sigma_model.select_dates = select_dates.__get__(two_sigma_model)
two_sigma_model.select_dates(datetime.datetime(2009, 1, 1, 0, 0, 0, 0, pytz.UTC))


# In[ ]:


gc.collect()
show_mem_usage()


# <a id="4"></a> <br>
# ### 1.2 Feature engineering
# A first step consists in performing some basic feature engineering:

# In[ ]:


@timeit
def column_combinations(self):
    self.market['price_diff'] = self.market['close'] - self.market['open']

@timeit
def asset_codes_encoding(self):
    self.news['assetCodesLen'] = self.news['assetCodes'].map(lambda x: len(eval(x)))
    self.news['assetCodes'] = self.news['assetCodes'].str.findall(f"'([\w\./]+)'").apply(lambda x: x[0])

@timeit
def news_count(self):
    t = self.news.groupby(['time', 'assetName']).size().reset_index(name='news_count')
    self.news = pd.merge(self.news, t, on=['time', 'assetName'])
#______________________________________________________________________
two_sigma_model.column_combinations = column_combinations.__get__(two_sigma_model)
two_sigma_model.asset_codes_encoding = asset_codes_encoding.__get__(two_sigma_model)
two_sigma_model.news_count = news_count.__get__(two_sigma_model)

two_sigma_model.column_combinations()
two_sigma_model.asset_codes_encoding()
two_sigma_model.news_count()    


# <a id="5"></a> <br>
# #### 1.2.1 Engineering  the `News` data
# The `news` data contains a few numerical columns. For each instrument and each day, we aggregate the corresponding variables with common statistics:

# In[ ]:


@timeit
def aggregate_numericals(self):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    self.news_numerical = self.news.select_dtypes(include=numerics).copy()
    self.news_numerical_columns = self.news_numerical.columns
    self.news_numerical.loc[:, 'assetCodes'] = self.news['assetCodes'] 
    self.news_numerical.loc[:, 'time'] = self.news['time']

    agg_func = {
        'takeSequence': ['sum', 'mean', 'max', 'min', 'std'],
        'bodySize': ['sum', 'mean', 'max', 'min', 'std'],
        'marketCommentary': ['sum', 'mean'],
        'sentenceCount': ['sum', 'mean', 'max', 'min', 'std'],
        'wordCount': ['sum', 'mean', 'max', 'min', 'std'],
        'relevance': ['sum', 'mean', 'max', 'min', 'std'],
        'firstMentionSentence': ['sum', 'mean', 'max', 'min', 'std'],
        'sentimentNegative': ['sum', 'mean', 'max', 'min', 'std'],
        'sentimentNeutral': ['sum', 'mean', 'max', 'min', 'std'],
        'sentimentPositive': ['sum', 'mean', 'max', 'min', 'std'],
        'sentimentWordCount': ['sum', 'mean', 'max', 'min', 'std'],
        'noveltyCount12H': ['sum', 'mean', 'max', 'min', 'std'],
        'noveltyCount24H': ['sum', 'mean', 'max', 'min', 'std'],
        'noveltyCount3D': ['sum', 'mean', 'max', 'min', 'std'],
        'noveltyCount5D': ['sum', 'mean', 'max', 'min', 'std'],
        'noveltyCount7D': ['sum', 'mean', 'max', 'min', 'std'],
        'volumeCounts12H': ['sum', 'mean', 'max', 'min', 'std'],
        'volumeCounts24H': ['sum', 'mean', 'max', 'min', 'std'],
        'volumeCounts3D': ['sum', 'mean', 'max', 'min', 'std'],
        'volumeCounts5D': ['sum', 'mean', 'max', 'min', 'std'],
        'volumeCounts7D': ['sum', 'mean', 'max', 'min', 'std'],
        'news_count': ['max']
        }

    self.news_numerical = self.news_numerical.groupby(['assetCodes', 'time']).agg(agg_func)
    self.news_numerical.columns = ['_'.join(col).strip()
                                   for col in self.news_numerical.columns.values]
    self.news_numerical.reset_index(inplace=True)
#______________________________________________________________________
two_sigma_model.aggregate_numericals = aggregate_numericals.__get__(two_sigma_model)
#two_sigma_model.aggregate_numericals() 


# Concerning the categorical columns, we only consider the last entry when for an instrument, there are multiple entries on a single day:

# In[ ]:


@timeit
def aggregate_categoricals(self): 
    categ_columns = [col for col in self.news.columns
                     if col not in self.news_numerical_columns]
    if self.verbose:
        print("news's categorical columns:\n", categ_columns)
    temp = self.news[self.news['news_count'] > 1][categ_columns].copy()
    multiple_articles = temp.groupby(['assetCodes', 'time']).tail(1)
    single_articles = self.news[self.news['news_count'] == 1][categ_columns]
    self.news_categ = pd.concat([multiple_articles, single_articles])
#________________________________________________________________________________________
two_sigma_model.aggregate_categoricals = aggregate_categoricals.__get__(two_sigma_model)
#two_sigma_model.aggregate_categoricals() 


# <a id="6"></a> <br>
# #### 1.2.2 merging the `News` and `Market` data

# In[ ]:


@timeit
def merge_news(self):
    self.news = pd.merge(self.news_numerical,
                         self.news_categ,
                         on = ['assetCodes', 'time'],
                         how='left')
    # freeing memory
    self.news_categ = 0
    self.news_numerical = 0

@timeit
def merge_market_news(self, keep_news=False):
    if keep_news:
        self.merged_df = pd.merge(self.market,
                                  self.news,
                                  left_on=['time', 'assetCode'],
                                  right_on=['time', 'assetCodes'],
                                  how='left')
    else:
        self.merged_df = self.market
    # freeing memory
    self.news = 0
    self.market = 0
    self.merged_df = reduce_mem_usage(self.merged_df, self.verbose)
    if self.verbose:
        print("merged_df's shape is: {}".format(self.merged_df.shape))
#_________________________________________________________________________
two_sigma_model.merge_news = merge_news.__get__(two_sigma_model)
two_sigma_model.merge_market_news = merge_market_news.__get__(two_sigma_model)

#two_sigma_model.merge_news()
two_sigma_model.merge_market_news() 


# In[ ]:


gc.collect()
show_mem_usage()


# <a id="7"></a> <br>
# ####  1.2.3 Indexing categorical columns
# At this stage, we define the categorical columns we will subsequently introduce in LGBM as `categorical_features`. These index is set during training and then reloaded during the prediction stage:

# In[ ]:


@timeit
def set_labels(self, labeled_columns, define_index=False):
    #____________________________________________________
    # indexation during training stage
    if define_index:
        self.indexer = {}
        for col in labeled_columns:
            _, self.indexer[col] = pd.factorize(self.merged_df[col])
    # label encoding
    self.categorical_columns = labeled_columns
    if self.verbose:
        print("categorical variables: {}".format(labeled_columns))
    for col in labeled_columns:
        self.merged_df[col] = self.indexer[col].get_indexer(self.merged_df[col])
#_____________________________________________________________________________
two_sigma_model.set_labels = set_labels.__get__(two_sigma_model)
#two_sigma_model.set_labels(['headlineTag', 'provider'], define_index=True)    
two_sigma_model.set_labels([], define_index=True)    


# <a id="8"></a> <br>
# #### 1.2.4 Lagged information
# A way to introduce the time is to create variables that contain past information (either time averages or lagged quantities). Below, we consider the returns on the `close` and `open` variables with respect to past values:

# In[ ]:


@timeit
def save_history(self):
    self.common_lag_features = ['time', 'assetCode'] 
    self.used_for_news_lag = [] # ['sentimentNegative_mean']
    self.used_for_lag = ['price_diff', 'close']
    self.history_df =  self.merged_df[self.common_lag_features +
                                      self.used_for_lag +
                                      self.used_for_news_lag].copy()
#_____________________________________________________________________________
two_sigma_model.save_history = save_history.__get__(two_sigma_model)
two_sigma_model.save_history()    


# In[ ]:


gc.collect()
show_mem_usage()


# In[ ]:


def run_parallel(grouped, method, verbose):
    with parallel.Parallel(n_jobs=-1, verbose=verbose) as par:
        segments = par(delayed(method)(df_seg) for df_seg in grouped)
    return segments

def account_for_past(df_code, n_lag=[7, 14, 21], shift_size=1):
    features = [c for c in df_code.columns if c not in ['assetCode']]
    lag_columns = []
    for col in features:
        for window in n_lag:
            if LAST_DATE_ONLY:
                tmp = df_code[df_code.index.max() - 2*timedelta(window):df_code.index.max()]
                rolled = tmp[col].shift(shift_size, freq='D').rolling(window=window)
            else:
                rolled = df_code[col].shift(shift_size, freq='D').rolling(window=window)
                
#             for fct in [np.mean, min, max]:
#                 col_name = '{}_{}_past{}'.format(col, fct.__name__, window)
#                 lag_columns.append(col_name)
#                 df_code[col_name] = rolled.apply(fct)

                col_name = '{}_median_past{}'.format(col, window)
                lag_columns.append(col_name)
                df_code[col_name] = rolled.median()
            
                col_name = '{}_max_past{}'.format(col, window)
                lag_columns.append(col_name)
                df_code[col_name] = rolled.max()
                
                col_name = '{}_min_past{}'.format(col, window)
                lag_columns.append(col_name)
                df_code[col_name] = rolled.min()
                
    return lag_columns, df_code.drop(features, axis=1)

def create_returns(df_code, n_days=[3, 5, 7, 12]):
    features = [c for c in df_code.columns if c not in ['assetCode']]
    lag_columns = []
    for col in features:
        for days in n_days:
            col_shift = 'shift_{}_{}'.format(col, days)  
            col_name = 'returns_{}_PrevRaw{}'.format(col, days)
            lag_columns.append(col_name)
            df_code.loc[:, col_shift] = df_code[col].shift(days, freq='D')
            df_code[col_name] = (df_code[col_shift] - df_code[col]) / df_code[col_shift]
            df_code.drop(col_shift, axis=1, inplace=True)
    return lag_columns, df_code.drop(features, axis=1)

@timeit
def define_lagged_var(self):
    start_time = time.time()
    list_df = [d[1][self.common_lag_features + ['price_diff']].set_index('time').copy()
               for d in self.history_df.groupby('assetCode', sort=False)]
    if self.verbose:
        print('assetCodes: {}, time to group: {:<5.2f}s'.format(len(list_df), time.time() - start_time))
        
    grouped = run_parallel(list_df, account_for_past, int(self.verbose))
    self.lag_columns = grouped[0][0]
    self.merged_df = pd.merge(self.merged_df,
                              pd.concat([d[1] for d in grouped]).reset_index(),
                              on=self.common_lag_features,
                              how='left')
    
@timeit
def define_lagged_return(self):
    start_time = time.time()
    list_df = [d[1][self.common_lag_features + ['close']].set_index('time').copy()
               for d in self.history_df.groupby('assetCode', sort=False)]
    if self.verbose:
        print('assetCodes: {}, time to group: {:<5.2f}s'.format(len(list_df), time.time() - start_time))
        
    grouped = run_parallel(list_df, create_returns, int(self.verbose))
    self.lag_columns = grouped[0][0]
    self.merged_df = pd.merge(self.merged_df,
                              pd.concat([d[1] for d in grouped]).reset_index(),
                              on=self.common_lag_features,
                              how='left')
        
    
#_____________________________________________________________________________
two_sigma_model.define_lagged_var = define_lagged_var.__get__(two_sigma_model)
two_sigma_model.define_lagged_return = define_lagged_return.__get__(two_sigma_model)
LAST_DATE_ONLY = False
two_sigma_model.define_lagged_var()      
two_sigma_model.define_lagged_return()      


# In[ ]:


@timeit
def news_moving_averages(self, col, window):
    col_name = 'news_mva_'+ col + '_' + str(window) + 'days'
    if self.verbose:
        print("column '{}' has been created".format(col_name))
    avg_col =         self.history_df[self.common_lag_features+[col]].set_index('time').        groupby('assetCode').rolling(window=window, freq='D').mean().reset_index()
    avg_col.rename(columns = {col: col_name}, inplace = True)
    self.merged_df = pd.merge(self.merged_df,
                              avg_col[self.common_lag_features + [col_name]],
                              on=self.common_lag_features,
                              how='left')
    return col_name

@timeit
def calc_news_moving_averages(self):    
    self.news_lag_columns = []
    for col in ['sentimentNegative_mean']:
        new_col = self.news_moving_averages(col, 7)
        self.news_lag_columns.append(new_col)
#_____________________________________________________________________________
two_sigma_model.news_moving_averages = news_moving_averages.__get__(two_sigma_model)
two_sigma_model.calc_news_moving_averages = calc_news_moving_averages.__get__(two_sigma_model)
#two_sigma_model.calc_news_moving_averages()   


# <a id="9"></a> <br>
# #### 1.3 Feature selection
# Before training, we select the variables we want to keep in our model:

# In[ ]:


@timeit
def select_variables(self):
    removed_columns = [
        'assetCode', 'assetCodes', 'assetCodesLen', 'assetName_x', 'assetName_y', 'assetName',
        'audiences', 'firstCreated', 'headline', 'returnsOpenNextMktres10',
        'sourceId', 'subjects', 'time', 'universe','sourceTimestamp']
    self.selected_variables = [c for c in self.merged_df.columns if c not in removed_columns]
    
    if self.verbose:
        print("variables kept: {}".format(self.selected_variables))
#_____________________________________________________________________________
two_sigma_model.select_variables = select_variables.__get__(two_sigma_model)
two_sigma_model.select_variables()  


# <a id="10"></a> <br>
# ## 2. Training the model
# 
# As a first step, we split the data set in order to cross-validate the model according to time:

# In[ ]:


@timeit
def define_time_cv(self):
    X = self.merged_df[self.selected_variables]
    target = self.merged_df['returnsOpenNextMktres10']

    time = self.merged_df['time']
    universe = self.merged_df['universe']

    n_train = int(X.shape[0] * 0.8)
    self.X_train, self.y_train = X.iloc[:n_train], target[:n_train]
    self.X_valid, self.y_valid = X.iloc[n_train:], target[n_train:]
    self.t_valid = time.iloc[n_train:]

    # For valid data, keep only those with universe > 0. This will help calculate the metric
    u_valid = (universe.iloc[n_train:] > 0)
    self.t_valid = time.iloc[n_train:]

    self.X_valid = self.X_valid[u_valid]
    self.y_valid = self.y_valid[u_valid]
    self.t_valid = self.t_valid[u_valid]
    del u_valid
#________________________________________________________________________________
two_sigma_model.define_time_cv = define_time_cv.__get__(two_sigma_model)
two_sigma_model.define_time_cv()  


# and then prepare the data for LGBM:

# In[ ]:


@timeit
def prep_lgbm_data(self):
    
    self.dtrain = lgb.Dataset(
        self.X_train.values, self.y_train,
        feature_name = self.selected_variables,
        categorical_feature = self.categorical_columns,
        free_raw_data = False)
    
    self.dvalid = lgb.Dataset(
        self.X_valid.values, self.y_valid,
        feature_name = self.selected_variables,
        categorical_feature = self.categorical_columns,
        free_raw_data = False)
    
    self.dvalid.params = {'extra_time': self.t_valid.factorize()[0]}
#________________________________________________________________________________
two_sigma_model.prep_lgbm_data = prep_lgbm_data.__get__(two_sigma_model)
two_sigma_model.prep_lgbm_data() 


# We define a function that encodes the competition's metric:

# In[ ]:


def sigma_score(preds, valid_data):
    df_time = valid_data.params['extra_time']
    labels = valid_data.get_label()
    val = pd.DataFrame()
    val['time'] = df_time
    val['y'] = preds * labels.values
    output = val.groupby('time').sum()
    score = output['y'].mean() / output['y'].std()
    return 'sigma_score', score, True


# to finally train the model:

# In[ ]:


def clean_frames(self):
    del self.merged_df
#________________________________________________________________________________
two_sigma_model.clean_frames = clean_frames.__get__(two_sigma_model)
two_sigma_model.clean_frames()


# In[ ]:


two_sigma_model.X_train[-15:]


# In[ ]:


gc.collect()
show_mem_usage()


# In[ ]:


@timeit
def train_model(self, lgb_params):
    evals_result = {}
    self.model = lgb.train(
        lgb_params,
        self.dtrain,
        num_boost_round= 10000,
        valid_sets=(self.dvalid,),
        valid_names=('valid',),
        verbose_eval=100,
        early_stopping_rounds=200,
        feval=sigma_score,
        evals_result=evals_result
    )
    df_result = pd.DataFrame(evals_result['valid'])
    self.num_boost_round, valid_score =         df_result['sigma_score'].idxmax()+1, df_result['sigma_score'].max()
    
    print(f'Best score was {valid_score:.5f} on round {self.num_boost_round}')
    
    del self.X_train
    del self.y_train
    del self.X_valid
    del self.y_valid
#________________________________________________________________________________
two_sigma_model.train_model = train_model.__get__(two_sigma_model)

lgb_params = dict(
    objective = 'regression_l1',
    learning_rate = 0.01,
    num_leaves = 51,
    max_depth = 8,
    bagging_fraction = 0.9,
    bagging_freq = 1,
    feature_fraction = 0.9,
    lambda_l1 = 0.0,
    lambda_l2 = 1.0,
    metric = 'None', 
    seed = 42)


two_sigma_model.train_model(lgb_params) 


# We can inspect the feature importance:

# In[ ]:


# two_sigma_model.model.feature_name()


# In[ ]:


# liste = list(zip(two_sigma_model.model.feature_name(),
#     two_sigma_model.model.feature_importance('gain')))
# liste.sort(key = lambda x:x[1], reverse=True)
# [x[0] for x in liste[:50]]


# In[ ]:


def feat_importances(self):
    fig, ax = plt.subplots(1, 1, figsize=(11, 20))
    lgb.plot_importance(self.model, ax, importance_type='gain')
    fig.tight_layout()
#________________________________________________________________________________
two_sigma_model.feat_importances = feat_importances.__get__(two_sigma_model)
two_sigma_model.feat_importances()


# <a id="11"></a> <br>
# ## 3. Model pipeline
# During the prediction stage, the observations of each day are given one after the other and we have to make the prediction before loading the data of the next day.
# Hence, in order to perform time averages or calculate lagged values, we have to enrich our data while it comes. We then define a method that concatentate the whole process described above:

# In[ ]:


@timeit
def data_prep(self, market, news):
    self.verbose = False
    self.market = market
    self.news = news
    self._format_dates()
    self._convert_booleans()
    self.column_combinations()
    self.asset_codes_encoding()
    self.news_count()   
    #self.aggregate_numericals()
    #self.aggregate_categoricals() 
    #self.merge_news()
    self.merge_market_news()
    #self.set_labels(['headlineTag', 'provider'])  
    self.set_labels([])  
#________________________________________________________________________________
two_sigma_model.data_prep = data_prep.__get__(two_sigma_model)


# <a id="12"></a> <br>
# ## 4. Test set predictions

# In[ ]:


days = env.get_prediction_days()


# Finally, we define a method that adds a new day to the data currently available:

# In[ ]:


@timeit
def add_new_date(self):
    self.history_df = pd.concat([self.history_df,
                                 self.merged_df[
                                      self.common_lag_features +
                                      self.used_for_lag +
                                      self.used_for_news_lag]])
    if self.verbose:
        print("data for lagged quantities from {} to {}".format(
            self.lag_df['time'].min().date(),
            self.lag_df['time'].max().date()))
#_______________________________________________________________
two_sigma_model.add_new_date = add_new_date.__get__(two_sigma_model)


# and a method that performs the predictions:

# In[ ]:


@timeit
def predict(self, date, pred_template):
    
    df = self.merged_df[self.merged_df['time'] == date]
    
    predictions = self.model.predict(
        df[self.selected_variables].values,
        ntree_limit = self.num_boost_round
    )
    
    preds = pd.DataFrame({'assetCode': df['assetCode'],
                          'confidence': np.clip(predictions, -1, 1)})
    
    self.formated_pred =         pred_template.merge(preds, how='left').        drop('confidenceValue', axis=1).        fillna(0).        rename(columns={'confidence': 'confidenceValue'})
#_______________________________________________________________
two_sigma_model.predict = predict.__get__(two_sigma_model)


# We are then able to loop over the test set days and create our prediction: 

# In[ ]:


two_sigma_model.history_df = two_sigma_model.history_df[two_sigma_model.history_df['time'] > (two_sigma_model.history_df['time'].max() - timedelta(30))]


# In[ ]:


TIME_FUNCTIONS = False
n_days = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    start_time = time.time()
    print(n_days, end=' ')
    two_sigma_model.data_prep(market_obs_df, news_obs_df)
    date = two_sigma_model.merged_df['time'].max()
    #_________________________________
    two_sigma_model.add_new_date()
    two_sigma_model.define_lagged_var()
    two_sigma_model.define_lagged_return()
    #two_sigma_model.calc_news_moving_averages()
    two_sigma_model.select_variables()
    two_sigma_model.predict(date, predictions_template_df)
    env.predict(two_sigma_model.formated_pred)


# In[ ]:


env.write_submission_file()


# In[ ]:





# In[ ]:




