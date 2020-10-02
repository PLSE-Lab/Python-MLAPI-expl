#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Daily Counts Model (Northquay)

# In[ ]:





# ### Run Settings

# In[ ]:


PRIVATE = False


# In[ ]:





# ### Import and Setup

# In[ ]:


import pandas as pd
import numpy as np
import os, sys, gc
import psutil


# In[ ]:


from collections import Counter
from random import shuffle
import math
from scipy.stats.mstats import gmean
import datetime


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns

pd.options.display.float_format = '{:.8}'.format
plt.rcParams["figure.figsize"] = (17, 5.5)
# %config InlineBackend.figure_format='retina'
 
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


pd.options.display.max_rows = 100


# In[ ]:


def ramCheck():
    print("{:.1f} GB used".format(psutil.virtual_memory().used/1e9 - 0.7))


# In[ ]:


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def memCheck():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


# In[ ]:


ramCheck()


# ### Load

# In[ ]:


stack_path = '/kaggle/input/c19week5stacks'
stacks = []
files = os.listdir(stack_path)
list.sort(files)
print(files)


# In[ ]:



for file in files:
    df = pd.read_csv(stack_path + '/' +file, index_col = 'ForecastId_Quantile')
    df.rename(columns = {'TargetValue':"preds_{}".format(file)}, inplace=True)
    stacks.append(df)
N_STACKS = len(stacks)


# In[ ]:


stack = stacks[0]
for new_stack in stacks[1:]:
    stack = stack.merge(new_stack, left_index=True, right_index=True)


# In[ ]:


stack.iloc[:, :N_STACKS]

stack[stack.sum]
# ### Train/Test

# In[ ]:


# path = '/kaggle/input/c19week3'
input_path = '/kaggle/input/covid19-global-forecasting-week-5'

# %% [code]
train = pd.read_csv(input_path + '/train.csv')
test = pd.read_csv(input_path  + '/test.csv')
sub = pd.read_csv(input_path + '/submission.csv')
example_sub = sub


# In[ ]:


train.rename(columns={'Country_Region': 'Country'}, inplace=True)
test.rename(columns={'Country_Region': 'Country'}, inplace=True)
# sup_data.rename(columns={'Country_Region': 'Country'}, inplace=True)


# In[ ]:


train['Place'] = train.Country + ('_' + train.Province_State).fillna("") +  ('_' + train.County ).fillna("")
test['Place'] = test.Country +  ('_' + test.Province_State).fillna("") + ('_' + test.County).fillna("") 


# In[ ]:


train.Place.unique()

train.Place.unique()[::1000]
# In[ ]:


train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)


# In[ ]:


train_bk = train.copy()
full_train = train.copy()

if PRIVATE:
    test = test[ pd.to_datetime(test.Date) >  train.Date.max()]
    pp = 'private'
else:
    pp = 'public'

train = train[train.Date < test.Date.min()]
# In[ ]:





# ### Join with Test

# In[ ]:


pdt = ['Place', 'Date', 'Target']


# In[ ]:


test.head()


# In[ ]:


stack['ForecastId'] = stack.index.str.split('_').str[0].astype(int)
stack['Quantile'] = stack.index.str.split('_').str[1]

stack[stack.iloc[:]]
# In[ ]:





# In[ ]:


full_stack = stack.merge(test, on='ForecastId', validate='m:1')    .merge(train, on=pdt, suffixes=('', '_y'), how='left')


# In[ ]:


full_stack[full_stack.Place=='US']


# In[ ]:


np.median(full_stack.iloc[:,:N_STACKS], axis = 1)


# In[ ]:


non_zero = full_stack[np.percentile(full_stack.iloc[:,:N_STACKS], 20, axis = 1) != 0]

full_stack.iloc[:,:N_STACKS].corr()
# In[ ]:


non_zero.iloc[:,:N_STACKS].corr()

full_stack['Impact'] = full_stack['TargetValue'] * train['Weight']
# In[ ]:





# ### FEATURES

# In[ ]:


# every Country, Province_State, Quantile, Target, Population,


# In[ ]:


full_stack['elapsed'] = (full_stack.Date - full_stack.Date.min()).dt.days +1

full_stack.elapsed
# In[ ]:


preds = full_stack.iloc[:, :N_STACKS]


# In[ ]:


full_stack['mean_pred'] = np.mean(preds, axis = 1)
full_stack['stdev_pred'] = np.std(preds, axis = 1)
full_stack['mean_over_stdev_pred'] = (full_stack.mean_pred / full_stack.stdev_pred ).fillna(-1)

full_stack['mean_log_pred'] = np.mean( np.log( 1 + np.clip(preds, 0, None)), axis = 1)
full_stack['stdev_log_pred'] = np.std( np.log( 1 + np.clip(preds, 0, None)), axis = 1)
full_stack['mean_over_stdev_log_pred'] = ( full_stack.mean_log_pred / full_stack.stdev_log_pred ).fillna(-1)


# In[ ]:


full_stack


# In[ ]:





# ### Basics

# In[ ]:


pdt = ['Place', 'Date', 'Target']


# In[ ]:


population = train.groupby('Place').Population.mean()
state = train.groupby('Place').Province_State.first()


# In[ ]:


train_avg = train.groupby(['Place', 'Date', 'Target']).mean()


# In[ ]:





# In[ ]:


train_pivot_cc = train_avg[train_avg.index.get_level_values(2)=='ConfirmedCases'].reset_index()    .pivot('Date', 'Place', 'TargetValue')


# In[ ]:


train_pivot_f = train_avg[train_avg.index.get_level_values(2)=='Fatalities'].reset_index()    .pivot('Date', 'Place', 'TargetValue')


# In[ ]:





# ### Begin Basic Features

# In[ ]:


data_cc = train_pivot_cc
data_fs = train_pivot_f


# In[ ]:


def columize(pivot_df):
    return pivot_df.reset_index().melt(id_vars = 'Date', value_name = 'Value').Value


# ### Population Features

# In[ ]:


full_stack['place_type'] = full_stack.Place.str.split('_').apply(len)

max_county_population = train[~train.County.isnull()].groupby('CountryState').Population.max()dataset['largest_county'] = np.where(dataset.place_type >= 2, 
                                     np.log( 1+ dataset.Place.map(train.groupby('Place').CountryState.first())\
                                            .map(max_county_population) ), -10 )
dataset.largest_county.fillna(-15, inplace=True)dataset.groupby('Place').largest_county.first().sort_values().plot(kind='hist', bins = 250);dataset['largest_county_vs_state_population'] = \
    dataset.largest_county - dataset.state_population
dataset.largest_county_vs_state_population.plot(kind='hist', bins = 250);dataset.groupby('Place').largest_county_vs_state_population.first().sort_values()[::40]dataset.county_pct_state_population.plot(kind='hist', bins = 250);PF_NOISE = 0.0;
# In[ ]:





# In[ ]:


test.Date.min()


# In[ ]:


full_stack.Date.min()


# In[ ]:





# ### Calc Features

# In[ ]:


dataset = data_cc.astype(np.float32).reset_index().melt(id_vars='Date', value_name = 'ConfirmedCases')
dataset = dataset.merge(data_fs.astype(np.float32).reset_index()                            .melt(id_vars='Date', value_name = 'Fatalities'),
                        on = ['Date', 'Place'])


# In[ ]:


dataset = dataset[dataset.Date == dataset.Date.max()].iloc[:,:2]

datasetfor window in [2, 4, 7, 14, 21, 35, 63]:
    mp = int(np.ceil(window ** 0.7))
    # EWMs
    dataset['cc_ewm_{}d'.format(window)] = columize(data_cc.ewm(span = window).mean())
    dataset['fs_ewm_{}d'.format(window)] = columize(data_fs.ewm(span = window).mean())

    # means
    dataset['cc_mean_{}d'.format(window)] = columize(data_cc.rolling(window, mp).mean())
    dataset['fs_mean_{}d'.format(window)] = columize(data_fs.rolling(window, mp).mean())

    
    # zero-days
    if window < 15:
        dataset['cczero_ewm_{}d'.format(window)] = np.round(columize((data_cc == 0).ewm(span=window).mean()), 2)
        dataset['fszero_ewm_{}d'.format(window)] = np.round(columize((data_fs == 0).ewm(span=window).mean()), 2)
    
    if window >= 7:
        # medians
        dataset['cc_median_{}d'.format(window)] = columize(data_cc.rolling(window, mp).median())
        dataset['fs_median_{}d'.format(window)] = columize(data_fs.rolling(window, mp).median())

        # low-end percentiles
        dataset['cc_vlow_{}d'.format(window)] = columize(data_cc.rolling(window, mp).quantile(0.10))
        dataset['fs_vlow_{}d'.format(window)] = columize(data_fs.rolling(window, mp).quantile(0.10))

        dataset['cc_low_{}d'.format(window)] = columize(data_cc.rolling(window, mp).quantile(0.25))
        dataset['fs_low_{}d'.format(window)] = columize(data_fs.rolling(window, mp).quantile(0.25))

        # high-end percentiles
        dataset['cc_high_{}d'.format(window)] = columize(data_cc.rolling(window, mp).quantile(0.75))
        dataset['fs_high_{}d'.format(window)] = columize(data_fs.rolling(window, mp).quantile(0.75))

        dataset['cc_vhigh_{}d'.format(window)] = columize(data_cc.rolling(window, mp).quantile(0.90))
        dataset['fs_vhigh_{}d'.format(window)] = columize(data_fs.rolling(window, mp).quantile(0.90))
        

        # stdev
        dataset['cc_stdev_{}d'.format(window)] = columize(data_cc.rolling(window, mp).std())
        dataset['fs_stdev_{}d'.format(window)] = columize(data_fs.rolling(window, mp).std())
        
        # stdev / mean
        dataset['ccstdev_over_mean_{}d'.format(window)] = columize( data_cc.rolling(window, mp).std()
                                                                      / data_cc.rolling(window, mp).mean())
        dataset['fsstdev_over_mean_{}d'.format(window)] = columize( data_fs.rolling(window, mp).std()
                                                                      / data_fs.rolling(window, mp).std())
        
        # skewness
        dataset['ccskew_{}d'.format(window)] = columize(data_cc.rolling(window, mp).skew())
        dataset['fsskew_{}d'.format(window)] = columize(data_fs.rolling(window, mp).skew())

        # kurtosis
        dataset['cckurtosis_{}d'.format(window)] = columize(data_cc.rolling(window, mp).kurt())
        dataset['fskurtosis_{}d'.format(window)] = columize(data_fs.rolling(window, mp).kurt())
        


# In[ ]:


dataset


# In[ ]:


dataset.shape


# In[ ]:


CC = 'ConfirmedCases'
FS = 'Fatalities'


# In[ ]:


train_cc = train[train.Target==CC].sort_values(['Place', 'Date'])
train_fs = train[train.Target==FS].sort_values(['Place', 'Date'])


# In[ ]:


train_fs.tail()


# In[ ]:


train_cc.tail()


# In[ ]:





# In[ ]:


dataset


# In[ ]:





# In[ ]:


total_cc = data_cc.cumsum()
total_fs = data_fs.cumsum()

def rollDates(df, i, preserve=False):
    df = df.copy()
    if preserve:
        df['Date_i'] = df.Date
    df.Date = df.Date + datetime.timedelta(i)
    return df
# In[ ]:





# ### CFR

# In[ ]:


dataset['log_cfr'] = columize( (    (total_fs                                          + np.clip(0.015 * total_cc, 0, 0.3))                             / ( total_cc + 0.1) ).apply(np.log))


# In[ ]:


columize( (    (total_fs                                          + np.clip(0.015 * total_cc, 0, 0.3))                             / ( total_cc + 0.1) ).apply(np.log))


# In[ ]:





# In[ ]:


dataset

def cfr(case, fatality):
    cfr_calc = np.log(    (fatality \
                                         + np.clip(0.015 * case, 0, 0.3)) \
                            / ( case + 0.1) )
    return np.where(np.isnan(cfr_calc) | np.isinf(cfr_calc),
                           BLCFR, cfr_calc)# %% [code]
BLCFR = dataset[(dataset.days_since_first_case == 0)].log_cfr.median()
dataset.log_cfr.fillna(BLCFR, inplace=True)
dataset.log_cfr = np.where(dataset.log_cfr.isnull() | np.isinf(dataset.log_cfr),
                           BLCFR, dataset.log_cfr)
BLCFR
# In[ ]:


dataset.log_cfr

# %% [code]  ** SLOPPY but fine
dataset['log_cfr_3d_ewm'] = BLCFR + \
                (dataset.log_cfr - BLCFR).ewm(span = 3).mean()  \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
                     
dataset['log_cfr_8d_ewm'] = BLCFR + \
                (dataset.log_cfr - BLCFR).ewm(span = 8).mean()  \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/8, 0, 1), 2)

dataset['log_cfr_20d_ewm'] = BLCFR + \
                (dataset.log_cfr - BLCFR).ewm(span = 20).mean()  \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/20, 0, 1), 2)

dataset['log_cfr_3d_20d_ewm_crossover'] = dataset.log_cfr_3d_ewm - dataset.log_cfr_20d_ewm


# %% [code]
dataset.drop(columns = 'log_cfr', inplace=True)dataset.log_cfr_8d_ewm.plot(kind='hist', bins = 250);
# In[ ]:





# ### Per Capita vs. World and Similar Countries

# In[ ]:


date_total_cc = np.sum(total_cc, axis = 1)
date_total_fs = np.sum(total_fs, axis = 1)

populationdate_total_cc (total_cc + 1).apply(np.log)\
                                        - (population + 1).apply(np.log)\
                                   # %% [code]
dataset['ConfirmedCases_percapita_vs_world'] = columize( ( (total_cc + 1).apply(np.log)\
                                        - (population + 1).apply(np.log))\
                                     .subtract( (date_total_cc + 1).apply(np.log), axis = 'index')\
                                        + np.log(population.sum() + 1) )
                                        dataset.ConfirmedCases_percapita_vs_world.plot(kind='hist', bins = 250);
# In[ ]:





# ### Plots

# In[ ]:


data_cc['US'].plot();

data_cc['USWashington']train[train.Country=='US'].Province_State.unique()
# In[ ]:


US_cc = data_cc[train[(train.Country=='US') & (~train.Province_State.isnull()) &
              (train.County.isnull())].Place.dropna().unique()]
US_cc.plot(legend=False);
US_cc.ewm(span=10).mean().plot(legend=False);


# In[ ]:


(US_cc.ewm(span=10).mean() - US_cc.ewm(span=20).mean()).plot(legend=False);


# In[ ]:


(US_cc.rolling(10).mean() - US_cc.rolling(20).mean()).plot(legend=False);


# ### Explore

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # II. Modeling

#  ### Data Prep

# In[ ]:


full_stack


# In[ ]:





# In[ ]:




memCheck()
# In[ ]:


gc.collect()


# In[ ]:


ramCheck()


# In[ ]:




# Private vs. Public Training

if PRIVATE:
    data_test = data[ (data.Date_i == train.Date.max() ) & 
                     (data.Date.isin(test.Date.unique() ) ) ].copy()
else:
    MAX_PUBLIC = datetime.datetime(2020, 5, 11)
    data_test = data[ (data.Date_i == test.Date.min() - datetime.timedelta(1) ) & 
                     (data.Date.isin(test.Date.unique() ) ) &
                      (data.Date <= MAX_PUBLIC) ].copy()
# In[ ]:


ramCheck()

test.Date.min()# %% [code]
model_data =  data[ ( (len(test) == 0)  | (data.Date < test.Date.min()) )
                    &  (~data.ConfirmedCases_f.isnull())].copy()
del data
# In[ ]:


g = gc.collect();
ramCheck()

# Dates
test.Date.min()
model_data.Date.max()
model_data.Date_i.max()
[c for c in model_data.Place if '_' in c]model_data.tail()full_stack.TargetValuey_cases = model_data.ConfirmedCases_f
y_fatalities = model_data.Fatalities_f
places = model_data.Placegroup_dict = {}
for place in dataset.Place.unique():
    group_dict[place] = '_'.join(place.split('_')[0:2])
group_dict['US'] = 'US_New York'groups = model_data.Place.map(group_dict)CC_SCALING.sort_values(ascending=False)[:20]group_dictmodel_data.tail()# model_data = model_data[~( 
#                             ( np.random.rand(len(model_data)) < 0.8 )  &
#                           ( model_data.Country == 'China') &
#                               (model_data.Date < datetime.datetime(2020, 2, 15)) )]

# %% [code]
x_dates = model_data[['Date_i', 'Date', 'Place']]

# x_dates.rename({'Date': 'Date_f'}, inplace=True)

# %% [code]x = model_data.iloc[:, 6:].copy().drop(columns = ['Date_i'])
del model_datag = gc.collect();
ramCheck()data_test.Date.unique()

# %% [code]
test.Date.unique()x_test =  data_test[x.columns].copy()train.Date.max()
test.Date.max()x.tail()
# ### Model Setup

# In[ ]:


# %% [code]
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, PredefinedSplit, TimeSeriesSplit
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesRegressor
# from xgboost import XGBRegressor
# from sklearn.linear_model import HuberRegressor, ElasticNet
import lightgbm as lgb


# In[ ]:


def quantile_loss(true, pred, quantile = 0.5):
    loss = np.where(true >= pred, 
                        quantile*(true-pred),
                        (1-quantile)*(pred - true) )
    return np.mean(loss)   
    


# In[ ]:


def quantile_scorer(quantile = 0.5):
    return make_scorer(quantile_loss, False, quantile = quantile)

SEED = 3np.random.seed(SEED)enet_params = { 'alpha': [   3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,  ],
                'l1_ratio': [  0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.97, 0.99 ]}

et_params = {        'n_estimators': [50, 70, 100, 140],
                    'max_depth': [3, 5, 7, 8, 9, 10],
                      'min_samples_leaf': [30, 50, 70, 100, 130, 165, 200, 300, 600],
                     'max_features': [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                    'min_impurity_decrease': [0, 1e-5 ], #1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                    'bootstrap': [ True, False], # False is clearly worse          
                 #   'criterion': ['mae'],
                   }
# In[ ]:


lgb_params = {
                'max_depth': [  2, 3, 5, 7, 10, 12, 14, 16],
                'n_estimators': [ 50, 100, 150, 225, 350 ],   # continuous
                'min_split_gain': [0, 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1],
                'min_child_samples': [ 1, 2, 4, 7, 10, 13, 17, 22, 30, 40, 70, 100],
                'min_child_weight': [0], #, 1e-3],
                'num_leaves': [  5, 10, 20, 30, 50, 100],
                'learning_rate': [0.05, 0.07, 0.1],   #, 0.1],       
                'colsample_bytree': [0.1, 0.2, 0.33, 0.5, 0.65, 0.8, 0.9], 
                'colsample_bynode':[0.1, 0.2, 0.33, 0.5, 0.65, 0.81],
                'reg_lambda': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100,    ],
                'reg_alpha': [1e-5,  1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1, 1, 1, 10,  ], # 1, 10, 100, 1000, 10000],
                'subsample': [   0.9, 1],
                'subsample_freq': [1],
                'max_bin': [ 100, 125, 175, 255, 511],
               }    


# In[ ]:


lgb_quantile_params = {
                'max_depth': [1, 2, 3, 4, 5, 7, 10, 14],
                'n_estimators': [ 30, 70, 100, 150, 225, ],   # continuous
                'min_split_gain': [0, 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1],
                'min_child_samples': [ 1, 4, 7, 10, 20, 30, 40, 50, 70, 100, 120, 140, 200],
                'min_child_weight': [0], #, 1e-3],
                'num_leaves': [5, 10, 20, 30, 50],
                'learning_rate': [0.05, 0.07, 0.1],   #, 0.1],       
                'colsample_bytree': [0.1, 0.2, 0.33, 0.5, 0.65, 0.8, 0.9], 
                'colsample_bynode':[0.1, 0.2, 0.33, 0.5, 0.65, 0.81],
                'reg_lambda': [1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000,   ],
                'reg_alpha': [1e-5,  1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1, 1, 1, 10, ], # 1, 10, 100, 1000, 10000],
                'subsample': [   0.9, 1],
                'subsample_freq': [1],
                'max_bin': [  50, 90, 125, 175, 255, 511],
               }    


# In[ ]:


MSE = 'neg_mean_squared_error'
MAE = 'neg_mean_absolute_error'


# In[ ]:


def trainENet(x, y, groups, cv = 0, **kwargs):
    return trainModel(x, y, groups, 
                      clf = ElasticNet(normalize = True, selection = 'random', 
                                       max_iter = 3000),
                      params = enet_params, 
                      cv = cv, **kwargs)

def trainETR(x, y, groups, cv = 0, n_jobs = 5,  **kwargs):
    clf = ExtraTreesRegressor(n_jobs = 1)
    params = et_params
    return trainModel(x, y, groups, clf, params, cv, n_jobs, **kwargs)
# In[ ]:





# In[ ]:





# In[ ]:


def trainLGB(x, y, groups, cv = 0, n_jobs = -1, **kwargs):
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  objective = 'mae',
                            categorical_feature=CF
                      )
    print('created lgb regressor')
    params = lgb_params
    
#     categorical_feature=name:c1,c2,c3 
    
    return trainModel(x, y, groups, clf, params, cv, n_jobs,  **kwargs)


# In[ ]:


def trainLGBquantile(x, y, groups, cv = 0, n_jobs = -1, **kwargs):
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  objective = 'quantile', **kwargs,
                      )
    print('created lgb quantile regressor')
    params = lgb_quantile_params
    
    return trainModel(x, y, groups, clf, params, cv, n_jobs,  **kwargs)


# In[ ]:


def trainModel(x, y, groups, clf, params, cv = 0, n_jobs = None, 
                   verbose=0, splits=None, **kwargs):
        if n_jobs is None:
            n_jobs = -1
        folds = TimeSeriesSplit(n_splits = N_DATES)
        print('running randomized search')
        clf = RandomizedSearchCV(clf, params, 
                            cv=  folds, 
                                 n_iter = 12, 
                                verbose = 1, n_jobs = n_jobs, scoring = cv)
        f = clf.fit(x, y, groups)
        print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  
        
        best = clf.best_estimator_;  print(best)
        print("Best Score: {}".format(np.round(clf.best_score_,4)))
        
        return best


# In[ ]:





# In[ ]:


g = gc.collect()


# In[ ]:


ramCheck()


# In[ ]:


gc.collect()

x.tail().drop(columns = 
                    [c for c in data.columns if any(z in c for z in ['fs_', 'cc_'])]).iloc[:, 54:]
# In[ ]:





# In[ ]:


WEIGHTS  = train[train.Target == 'Fatalities'].groupby('Place').Weight.mean().sort_values()


# In[ ]:


WEIGHTS.head()


# In[ ]:


WEIGHTS.tail()


# In[ ]:


SCALE_RANGE = 2


# In[ ]:





def rescale_x(x, resample):
    x = x.copy()
    for col in RESAMPLE_COLS:
        x[col] = x[col] / resample
        
        
    return x


# In[ ]:


def runBags(x, y, groups, cv, bags = 3, model_type = trainLGB, 
            noise = 0.1, splits = None, weights = 1, resample = None, **kwargs):
    models = []
    for bag in range(bags):
        print("\nBAG {}".format(bag+1))

        # set size picked randomly
        ssr =  SET_FRAC * np.exp(  - SIZE_RANGE * np.random.rand()  )
        
        if weights is not None:
            ssr = ssr *   places.map(WEIGHTS)
            
        # weight by resample multiplier
        if resample is not None:
            ssr = ssr * resample;
            
        print('Max Subsample Wt: {}'.format(np.max(ssr)))
        
        # filter
        ss = ( np.random.rand(len(y)) < ssr  )
        
        
        print("n={}".format(ss.sum()))
        
        # change rescale at random -- for model smooth splits
        resample_ss = resample[ss]  /  np.clip(ssr[ss], 1, None)
        resample_ss = resample_ss * np.exp( SCALE_RANGE * ( np.random.rand(ss.sum()) - 0.5 ) )

        
#         print((resample_ss.sort_values(ascending = False))[::len(xss)//10])
#         print((y[ss].sort_values(ascending = False))[::ss.sum()//10])

        print("\nMean Count: {}\n".format(np.mean(y[ss])))
        
        xss = rescale_x(x[ss], resample_ss)
        yss = y[ss] / resample_ss
        print(yss.sum()); print()
        
#         print('Max Subsample Had Count of: {}'.format(yss[np.argmax(ssr)]))
        
        print("Largest Count from {}".format(groups[ss][yss.idxmax()]))
        print((yss.sort_values(ascending = False))[:10]); print()

        
        models.append(model_type(xss, yss, groups[ss], cv, **kwargs))
        return models

# %% [code]


# In[ ]:


BAG_MULT = 1

# x.shape


date_weights =  (1 * np.abs((x_dates.Date_i - test.Date.min()).dt.days) + 
                    2 * np.abs((x_dates.Date - test.Date.min()).dt.days))/3
                        
# In[ ]:


# NY's rise:
#   Largest County as Percent of State
#   Population of Largest County;
#   MUCH LARGER POPULATION SCALING--that sends even 10m people to a weight of 30 
#   (still leaves NY as very high per capita outlier though, and US as even bigger one...)
#    use population scaling to scale up counties--anything below 10k people is getting x30


# THE RISK: MODEL IS DOMINATED BY GETTING EARLY NYC RIGHT -- large % of error here?
#    helped by downscaling by population (population centers can boom very easy)
#    helped by not going 35 days out (too much)
#    helped by including largest outbreak so far (to show how quickly others of this pop ramped up)
#    helped by some indication country was about to take off (other states, etc.)
#    helped by flagging it as place with Largest County as Percent of State (NYC)
#    helped by flagging state as containing a huge County
#    others--worth understanding OOB error here, as may effect big countries;


# ### Data Cleanup

# In[ ]:


full_stack_clean = full_stack.drop(columns = ['ForecastId', 'Id', 'Weight'] + 
                        [c for c in full_stack if '_y' in c]  ).fillna('None')


# In[ ]:


UNIQUE_COLS = full_stack_clean.columns[full_stack_clean.iloc[:1000,:].groupby(['Place', 'Date']).nunique().mean() > 1]


# In[ ]:


UNIQUE_COLS


# In[ ]:


joined_data = full_stack_clean.groupby(['Place', 'Date']).first().reset_index()    [[c for c in full_stack_clean.columns if c not in UNIQUE_COLS]]


# In[ ]:


joined_data

full_stack_clean
# In[ ]:


UNIQUE_COLS = UNIQUE_COLS.drop(['Target', 'Quantile'])


# In[ ]:


for target in ['ConfirmedCases', 'Fatalities']:
    joined_data = joined_data.merge( full_stack[full_stack.Target == target]                                        .groupby(['Place', 'Date']).TargetValue.first().rename(target),
                                        on = ['Place', 'Date'])
    for quantile in full_stack.Quantile.unique():
        df = full_stack[ (full_stack.Target == target) 
                                                     & (full_stack.Quantile == quantile)]\
                                        [list(UNIQUE_COLS) + ['Date', 'Place']].drop(columns = 'TargetValue')
                                                
        df.columns = [ c if ((c=='Date') | (c== 'Place')) else
                              c + '_' + target[0].lower() + '_' + quantile 
                             for c in df.columns]
        joined_data = joined_data.merge(df ,
                                        on = ['Place', 'Date'])
        


# In[ ]:


joined_data.columns


# In[ ]:


joined_data = joined_data.sort_values(['Date', 'Place'])


# In[ ]:


assert (joined_data.isnull().sum() > 0).sum() <=2

joined_data.columnsscaler = joined_data.mean_pred_
# In[ ]:


for col in joined_data.columns[joined_data.dtypes == 'object']:
    joined_data[col] = joined_data[col].astype('category')

# joined_data['scale'] = 1


train_data = joined_data.dropna();
test_data = joined_data

x = train_data.drop(columns = ['ConfirmedCases', 'Fatalities', 'Date'])
x_test = test_data.drop(columns = ['ConfirmedCases', 'Fatalities', 'Date'])


dates = train_data.Date
y_cases = train_data.ConfirmedCases
y_fatalities = train_data.Fatalities
places = train_data.Place

full_stack.Quantile.unique()
# In[ ]:




def scale_x(x, s):
    x = x.copy()
    scaled_cols = [c for c in x.columns if 
                   (any(z in c for z in full_stack.Quantile.unique()) and 'over' not in c
                          and 'log' not in c) or ('scale' in c)]
    for col in scaled_cols:
        x[col] = x[col] / s
    return x
# In[ ]:


RESAMPLE_COLS = [c for c in x.columns if 
                   (any(z in c for z in full_stack.Quantile.unique()) and 'over' not in c
                          and 'log' not in c) or ('scale' in c)]

np.sqrt(273)
# In[ ]:


x['mean_pred_c_0.5'].max()


# In[ ]:


x['mean_pred_f_0.5'].max()


# In[ ]:


c_scale =  np.clip(x['mean_pred_c_0.5'] / 1000, 1/30, 30)
f_scale =  np.clip(x['mean_pred_f_0.5'] / 100, 1/30, 30)

# x_cases = scale_x(x, c_scale)
# x_fatalities = scale_x(x, c_scale)

# x_test_cases = scale_x(x_test, f_scale)
# x_test_fatalities = scale_x(x_test, f_scale)

x_cases.scale[::10]np.percentile(x_cases.scale, 0.1)c_scale.plot(kind='hist', bins = 250);
# In[ ]:




x.columnsx.columns.get_loc('elapsed')
# In[ ]:




x_test.Countryx.Country
# In[ ]:


x

x.columnsy_casesdates
# In[ ]:


N_DATES = len(dates.unique())
assert len(dates) % N_DATES == 0
print(N_DATES)


# In[ ]:


date_weights = 1


# In[ ]:





# In[ ]:


CF = "name:"  + ",".join(list(x.columns[x.dtypes == 'object']))
CF


# In[ ]:




x
# In[ ]:


SET_FRAC = 0.12
SIZE_RANGE = 0
BAGS = 1
SINGLE_MODEL = True


# #### Cases

# In[ ]:


lgb_c_clfs = []; lgb_c_noise = []


# In[ ]:


for iteration in range(0, int(math.ceil(1 * BAGS))):
    for noise in [ 0.05, 0.1, 0.2, 0.3, 0.4  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
#         cv_group = groups
        
        lgb_c_clfs.extend(runBags(x, y_cases, 
                          dates, #groups
                          MAE, num_bags, trainLGB, verbose = 0, 
                                          noise = noise, resample=c_scale
 
                                 ))
        lgb_c_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;


# ### Case Quantiles

# In[ ]:


lgb_c5_clfs = []; lgb_c5_noise = []
lgb_c95_clfs = []; lgb_c95_noise = []


# In[ ]:


alpha = 0.05
for iteration in range(0, int(math.ceil(1 * BAGS))):
    for noise in [ 0.05, 0.1,   ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;

        
        lgb_c5_clfs.extend(runBags(x, y_cases, 
                          dates, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise,  alpha = alpha, resample=c_scale

                                 ))
        lgb_c5_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;


# In[ ]:


alpha = 0.95
for iteration in range(0, int(math.ceil(1 * BAGS))):
    for noise in [ 0.05, 0.1, 0.2, 0.3, 0.4  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        
        
        lgb_c95_clfs.extend(runBags(x, y_cases, 
                          dates, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise,                                   alpha = alpha,
                                    resample=c_scale

                                 ))
        lgb_c95_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;


# In[ ]:





# ### Fatality Quantiles

# In[ ]:


lgb_f5_clfs = []; lgb_f5_noise = []
lgb_f95_clfs = []; lgb_f95_noise = []


# In[ ]:


alpha = 0.05
for iteration in range(0, int(math.ceil(1 * BAGS))):
    for noise in [ 0.05, 0.1,  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
#         cv_group = groups
        
        lgb_f5_clfs.extend(runBags(x, y_fatalities, 
                          dates, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise,
                                   alpha = alpha, resample =f_scale

                                 ))
        lgb_f5_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;


# In[ ]:


alpha = 0.95
for iteration in range(0, int(math.ceil(1 * BAGS))):
    for noise in [ 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
#         cv_group = groups
        
        lgb_f95_clfs.extend(runBags(x, y_fatalities, 
                          dates, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise,
                                   alpha = alpha, resample = f_scale

                                 ))
        lgb_f95_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;


# #### Fatalities

# In[ ]:


lgb_f_clfs = []; lgb_f_noise = []


# In[ ]:


for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in [  0.5,  1, 2, 3, 4, 5  ]:
        num_bags = 1 * int(np.ceil(np.sqrt(BAG_MULT)))
   
        lgb_f_clfs.extend(runBags(x, y_fatalities, 
                                  dates, #places, # groups, 
                                  MAE, num_bags, trainLGB, 
                                  verbose = 0, noise = noise, resample = f_scale

                                 ))
        lgb_f_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;


# #### CFR
lgb_cfr_clfs = []; lgb_cfr_noise = [];


for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in [    0.4, 1, 2, 3]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < 0.5 * PLACE_FRACTION :
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
 
        lgb_cfr_clfs.extend(runBags(x, y_cfr, 
                          cv_group, #groups
                          MSE, num_bags, trainLGB, verbose = 0, 
                                          noise = noise, 
                                          resample = date_weights

                                 ))
        lgb_cfr_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;
# ### MP: Model Parameters

# In[ ]:


lgb_c_clfs


# In[ ]:


lgb_f_clfs


# In[ ]:





# In[ ]:


lgb_c95_clfs


# In[ ]:


lgb_f95_clfs


# In[ ]:





# In[ ]:


lgb_c5_clfs


# In[ ]:


lgb_f5_clfs


# In[ ]:





# ### Feature Importance

# In[ ]:


def show_FI(model, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fis = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1][:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    


# In[ ]:


def avg_FI(all_clfs, featNames, featCount):
    # 1. Sum
    clfs = []
    for clf_set in all_clfs:
        for clf in clf_set:
            clfs.append(clf);
    print("{} classifiers".format(len(clfs)))
    fi = np.zeros( (len(clfs), len(clfs[0].feature_importances_)) )
    for idx, clf in enumerate(clfs):
        fi[idx, :] = clf.feature_importances_
    avg_fi = np.mean(fi, axis = 0)

    # 2. Plot
    fis = avg_fi
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1]#[:featCount]
    #print(indices)
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    
    return pd.Series(fis[indices], featNames[indices])


# In[ ]:


def linear_FI_plot(fi, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(np.absolute(fi))[::-1]#[:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fi[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    return pd.Series(fi[indices], featNames[indices])


# In[ ]:


preds.columns

x.columns
# In[ ]:


f = avg_FI([lgb_c_clfs], x.columns, 25)

for feat in preds.columns:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))
    
# for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
#              'world', 'continent', 'nearest', 'surrounding']:
#     print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# f[:100:3]

# print("{}: {:.2f}".format('sup_data', 
#                        f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
# print("{}: {:.2f}".format('contain_data', 
#                    f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))


# In[ ]:


f = avg_FI([lgb_f_clfs], x.columns, 25)

for feat in preds.columns:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))
    
# %% [code]
# for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
#             'world', 'continent', 'nearest', 'surrounding']:
#     print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# %% [code]
# print("{}: {:.2f}".format('sup_data', 
#                        f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
# print("{}: {:.2f}".format('contain_data', 
#                    f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))

x.columns

# In[ ]:





# In[ ]:


f = avg_FI([lgb_c95_clfs], x.columns, 25)

for feat in preds.columns:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))
    


# In[ ]:


f = avg_FI([lgb_f95_clfs], x.columns, 25)

for feat in preds.columns:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))
    


# In[ ]:





# In[ ]:


f = avg_FI([lgb_c5_clfs], x.columns, 25)

for feat in preds.columns:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))
    


# In[ ]:


f = avg_FI([lgb_f5_clfs], x.columns, 25)

for feat in preds.columns:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))
    

f = avg_FI([lgb_cfr_clfs], x.columns, 25)

# %% [code]
for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
            'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# %% [code]
# print("{}: {:.2f}".format('sup_data', 
#                        f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
# print("{}: {:.2f}".format('contain_data', 
#                    f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))
# ### Make Predictions

# In[ ]:


all_c_clfs = [lgb_c_clfs, ]#  enet_c_clfs]
all_f_clfs = [lgb_f_clfs] #, enet_f_clfs]
all_c5_clfs = [lgb_c5_clfs, ]#  enet_c_clfs]
all_f5_clfs = [lgb_f5_clfs] #, enet_f_clfs]
all_c95_clfs = [lgb_c95_clfs, ]#  enet_c_clfs]
all_f95_clfs = [lgb_f95_clfs] #, enet_f_clfs]



# all_cfr_clfs = [lgb_cfr_clfs]


# %% [code]
all_c_noise = [lgb_c_noise]
all_f_noise = [lgb_f_noise]
all_c5_noise = [lgb_c5_noise]
all_f5_noise = [lgb_f5_noise]
all_c95_noise = [lgb_c95_noise]
all_f95_noise = [lgb_f95_noise]


# all_cfr_noise = [lgb_cfr_noise]

# %% [code]


# In[ ]:


NUM_TEST_RUNS = 4

x_test = x
# In[ ]:


# %% [code]
c_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_c_clfs]), len(x_test)))
f_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_f_clfs]), len(x_test)))
c5_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_c5_clfs]), len(x_test)))
f5_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_f5_clfs]), len(x_test)))
c95_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_c95_clfs]), len(x_test)))
f95_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_f95_clfs]), len(x_test)))


# In[ ]:


def avg(x):
#     x = x[::2, :]
    return np.median(x, axis = 0)
#     return np.where( np.min(x,axis=0) > 0.1, gmean(np.clip(x, 0.1, None), axis = 0), np.median(x,axis = 0))
                    
#                     np.median(x,axis=0)
#     return (np.mean(x, axis=0) + np.median(x, axis=0))/2


# In[ ]:


def noisify(x, noise):
    return x


# In[ ]:


test_cc_scaling = 1; #data_test.Place.map(CC_SCALING) * LOW_COUNT_SCALING_TEST
test_fs_scaling = 1; #data_test.Place.map(FS_SCALING) * LOW_COUNT_SCALING_TEST

x_testrescale_x(x_test, test_cc_scaling)
# In[ ]:


xtr = np.exp( SCALE_RANGE * (np.random.rand(len(x_test)) -0.5 ) )


# In[ ]:


count = 0

for idx, clf in enumerate(lgb_c_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_c_noise[idx]
        np.random.shuffle(xtr)
        scaling = test_cc_scaling * xtr
        c_preds[count,:] = scaling * clf.predict(noisify(rescale_x(x_test, scaling), noise))
        count += 1


# In[ ]:


count = 0

for idx, clf in enumerate(lgb_c5_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_c5_noise[idx]
        np.random.shuffle(xtr)
        scaling = test_cc_scaling * xtr
        c5_preds[count,:] = scaling * clf.predict(noisify(rescale_x(x_test, scaling), noise))
        count += 1


# In[ ]:


count = 0

for idx, clf in enumerate(lgb_c95_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_c95_noise[idx]
        np.random.shuffle(xtr)
        scaling = test_cc_scaling * xtr
        c95_preds[count,:] = scaling * clf.predict(noisify(rescale_x(x_test, scaling), noise))
        count += 1


# In[ ]:


np.set_printoptions(precision = 3, suppress=True);


# In[ ]:


c_preds[:, :4]
c5_preds[:, :4]
c95_preds[:, :4]


# In[ ]:


count = 0

for idx, clf in enumerate(lgb_f_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_f_noise[idx]
        np.random.shuffle(xtr)
        scaling = test_fs_scaling * xtr
        f_preds[count,:] = scaling * clf.predict(noisify(rescale_x(x_test, scaling), noise))
        count += 1


# In[ ]:


count = 0

for idx, clf in enumerate(lgb_f5_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_f5_noise[idx]
        np.random.shuffle(xtr)
        scaling = test_fs_scaling * xtr
        f5_preds[count,:] = scaling * clf.predict(noisify(rescale_x(x_test, scaling), noise))
        count += 1


# In[ ]:


count = 0

for idx, clf in enumerate(lgb_f95_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_f95_noise[idx]
        np.random.shuffle(xtr)
        scaling = test_fs_scaling * xtr
        f95_preds[count,:] = scaling * clf.predict(noisify(rescale_x(x_test, scaling), noise))
        count += 1


# In[ ]:


f_preds[:, :4]
f5_preds[:, :4]
f95_preds[:, :4]


# In[ ]:





# In[ ]:


y_cases95_pred_blended_full = avg(c95_preds)
y_cases5_pred_blended_full = avg(c5_preds)
y_cases_pred_blended_full = avg(c_preds)
y_fatalities_pred_blended_full = avg(f_preds)
y_fatalities5_pred_blended_full = avg(f5_preds)
y_fatalities95_pred_blended_full = avg(f95_preds)


# In[ ]:




count = 0

for idx, clf in enumerate(lgb_cfr_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_cfr_noise[idx]
        cfr_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -10 , 10)
        count += 1def qPred(preds, pctile, simple=False):
    return np.mean(preds, axis = 0)
#     q = np.percentile(preds, pctile, axis = 0)
#     if simple:
#         return q;
#     resid = preds - q
#     resid_wtg = 2/100/len(preds)* ( np.clip(resid, 0, None) * (pctile) \
#                         + np.clip(resid, None, 0) * (100- pctile) )
#     adj = np.sum(resid_wtg, axis = 0)
#     return q + adjq = 50y_cases_pred_blended_full = qPred(c_preds, q) #avg(c_preds)
y_fatalities_pred_blended_full = qPred(f_preds, q) # avg(f_preds)
# y_cfr_pred_blended_full = qPred(cfr_preds, q) #avg(cfr_preds)c_preds.shape
f_preds.shape
# In[ ]:


if not SINGLE_MODEL:
    print(np.mean(np.corrcoef(c_preds[::NUM_TEST_RUNS]),axis=0))

    print(np.mean(np.corrcoef(f_preds[::NUM_TEST_RUNS]), axis=0))

    print(np.mean(np.corrcoef(c95_preds[::NUM_TEST_RUNS]),axis=0))

    print(np.mean(np.corrcoef(f95_preds[::NUM_TEST_RUNS]), axis=0))

# print(np.mean(np.corrcoef(cfr_preds[::NUM_TEST_RUNS]), axis = 0))


# In[ ]:


pd.Series(np.std(c_preds, axis = 0) / (100 + np.mean(c_preds, axis=0))).plot(kind='hist', bins = 250);


# In[ ]:


pd.Series(np.std(f_preds, axis = 0) / (10+ np.mean(f_preds, axis= 0))).plot(kind='hist', bins = 250);

# pd.Series(np.std(cfr_preds, axis = 0)).plot(kind='hist', bins = 50)

# %% [code]
pred = pd.DataFrame(np.hstack((np.transpose(c_preds),
                              np.transpose(f_preds))), index=x_test.index)
pred['Place'] = data_test.Place

pred['Date'] = data_test.Date_i
pred['Date_f'] = data_test.Date# %% [code]
pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][30: 60]# %% [code]
np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())], 2)[190:220:]data_cc['US_New York'].tail()
# In[ ]:




train[train.County=='New York'].Population.mean()pred.set_index('Place')[0].groupby('Place').max().sort_values()# %% [code] {"scrolled":false}
np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][220:-20],2)# %% [code]
c_preds.shape
x_test.shape
# In[ ]:





# ### Predict on Test Set
data_wp
# In[ ]:


data_test = test


# In[ ]:


data_test 


# In[ ]:


len(x_test)


# In[ ]:


len(y_cases_pred_blended_full)


# In[ ]:


joined_data


# In[ ]:




### %% [code]
data_wp = joined_data.copy()

data_wp['ConfirmedCases_pred'] = y_cases_pred_blended_full 
data_wp['Fatalities_pred'] = y_fatalities_pred_blended_full 
data_wp['ConfirmedCases5_pred'] = y_cases5_pred_blended_full 
data_wp['Fatalities5_pred'] = y_fatalities5_pred_blended_full 
data_wp['ConfirmedCases95_pred'] = y_cases95_pred_blended_full 
data_wp['Fatalities95_pred'] = y_fatalities95_pred_blended_full 

# data_wp['cfr_pred'] = y_cfr_pred_blended_fulldata_wpfull_stack.Quantile.unique()
# In[ ]:


q_labels = [ '5','', '95'];
data_rw = joined_data.copy()
for target in ['ConfirmedCases', 'Fatalities']:
    for idx, quantile in enumerate(full_stack.Quantile.unique()):
            data_rw[target+ q_labels[idx] + '_pred'] = 0
            

            data_rw[target+ q_labels[idx] + '_pred'] =                    data_rw[[      c + '_' + target[0].lower() + '_' + quantile 
                                     for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,-1] 


# In[ ]:


stack.iloc[:, :N_STACKS].iloc[:,:-1]


# In[ ]:


q_labels = [ '5','', '95'];
data_nq = joined_data.copy()
for target in ['ConfirmedCases', 'Fatalities']:
    for idx, quantile in enumerate(full_stack.Quantile.unique()):
            data_nq[target+ q_labels[idx] + '_pred'] = 0
            

            data_nq[target+ q_labels[idx] + '_pred']= np.median(              data_nq[[      c + '_' + target[0].lower() + '_' + quantile 
                                     for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,:-1], axis = 1)
           


# In[ ]:


q_labels = [ '5','', '95'];
data_wp = joined_data.copy()
for target in ['ConfirmedCases', 'Fatalities']:
    for idx, quantile in enumerate(full_stack.Quantile.unique()):
            data_wp[target+ q_labels[idx] + '_pred'] = 0
            

            data_wp[target+ q_labels[idx] + '_pred'] =                    data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
                                     for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,2] 

#             if q_labels[idx] == '5':
#                 data_wp[target+ q_labels[idx] + '_pred'] =\
#                         (    data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
#                                          for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,2] 
#                         +
#                         data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
#                                          for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,0] 
#                         )/2

                    
                    
#             if q_labels[idx] =='':
#                 data_wp[target+ q_labels[idx] + '_pred'] =\
#                     (    data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
#                                      for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,1] 
#                     +
#                     data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
#                                      for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,0] 
#                     )/2

                    
                    
#             if q_labels[idx] =='95':
#                 data_wp[target+ q_labels[idx] + '_pred'] =\
#                     data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
#                                      for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,2] 
                    
#                         np.mean(data_wp[[      c + '_' + target[0].lower() + '_' + quantile 
#                              for c in stack.iloc[:, :N_STACKS].columns]] , axis = 1)

data_wpdata_wp.groupby(['Place', 'Date'])[ ['ConfirmedCases', 'Fatalities'] + list(data_wp.columns[-6:])].first()
# In[ ]:




### %% [code]
data_wp = joined_data.copy()

data_wp['ConfirmedCases_pred'] = y_cases_pred_blended_full 
data_wp['Fatalities_pred'] = y_fatalities_pred_blended_full 
data_wp['ConfirmedCases5_pred'] = y_cases5_pred_blended_full 
data_wp['Fatalities5_pred'] = y_fatalities5_pred_blended_full 
data_wp['ConfirmedCases95_pred'] = y_cases95_pred_blended_full 
data_wp['Fatalities95_pred'] = y_fatalities95_pred_blended_full 

# data_wp['cfr_pred'] = y_cfr_pred_blended_fulllen(data_test)
len(y_cases_pred_blended_full)data_wp.Place.map(test)for f in ['ConfirmedCases', 'Fatalities']:
    data_wp[[f +'_f', f + '_pred']].corr()data_wp\
    [['Date', 'Place'] + 
            [c for c in data_wp if 
         any(z in c for z in ['Confirmed', 'Fatal'])
            and 'capita' not in c]].iloc[:40]
# In[ ]:


train.Date.max()
test.Date.min()


# In[ ]:


if len(test) > 0:
    base_date = test.Date.min() - datetime.timedelta(1)
else:
    base_date = train.Date.max()


# In[ ]:


base_date

stack.columns[0]plt.scatter(data_wp[stack.columns[0] + '_c_0.5'],data_wp.ConfirmedCases_pred,s=.1);
plt.xlim(0, 6000)
plt.ylim(0, 2000)plt.scatter(
        1 + np.clip(data_wp[stack.columns[0] + '_c_0.5'], 0, None),
            1 + np.clip(data_wp.ConfirmedCases_pred, 0, None), 
    s=.1);
plt.yscale('log')
plt.xscale('log')

plt.xlim(0, 6000)
plt.ylim(0, 2000)data_cc['US'].tail()data_wp[data_wp.Place=='US'].set_index('Date').iloc[:, -6:]
# In[ ]:




data_wp.columns
# In[ ]:


test_wp = pd.merge(test, data_wp[['Date', 'Place', 'ConfirmedCases_pred', 'Fatalities_pred',
                                      'ConfirmedCases5_pred', 'Fatalities5_pred',
                                      'ConfirmedCases95_pred', 'Fatalities95_pred',
                                    'elapsed']], 
            how='left', on = ['Date', 'Place'])


# In[ ]:


test_nq = pd.merge(test, data_nq[['Date', 'Place', 'ConfirmedCases_pred', 'Fatalities_pred',
                                      'ConfirmedCases5_pred', 'Fatalities5_pred',
                                      'ConfirmedCases95_pred', 'Fatalities95_pred',
                                    'elapsed']], 
            how='left', on = ['Date', 'Place'])


# In[ ]:


test_rw = pd.merge(test, data_rw[['Date', 'Place', 'ConfirmedCases_pred', 'Fatalities_pred',
                                      'ConfirmedCases5_pred', 'Fatalities5_pred',
                                      'ConfirmedCases95_pred', 'Fatalities95_pred',
                                    'elapsed']], 
            how='left', on = ['Date', 'Place'])

test_rw == test_nqtest_test_wp.tail()
# In[ ]:


data_wp.to_csv('data_wp.csv', index=False)
test_wp.to_csv('data_wp.csv', index=False)


# In[ ]:


subs = []


# In[ ]:


# test_wp['TargetValue'] = 
subs.append(pd.Series(np.where(test_wp.Target=='ConfirmedCases', test_wp.ConfirmedCases_pred,
                                                                    test_wp.Fatalities_pred), 
                      index = test_wp.ForecastId.astype(str) + '_0.5', copy=True))
# print(test_wp.TargetValue)

# test_wp['TargetValue'] = 
subs.append(pd.Series(np.where(test_wp.Target=='ConfirmedCases', test_wp.ConfirmedCases5_pred,
                                                                    test_wp.Fatalities5_pred), 
                      index = test_wp.ForecastId.astype(str) + '_0.05', copy=True ))
# print(test_wp.TargetValue)

# test_wp['TargetValue'] = 
subs.append(pd.Series(np.where(test_wp.Target=='ConfirmedCases', test_wp.ConfirmedCases95_pred,
                                                                    test_wp.Fatalities95_pred), 
                      index = test_wp.ForecastId.astype(str) + '_0.95', copy=True ))
# print(test_wp.TargetValue)


# In[ ]:


subs = pd.concat(subs).rename('TargetValue')


# In[ ]:


sub.sort_values('ForecastId_Quantile')


# In[ ]:


sub = pd.DataFrame(subs).reset_index().fillna(0).rename(columns={'ForecastId': 'ForecastId_Quantile'})


# In[ ]:


sub = example_sub[['ForecastId_Quantile']].merge(sub, on = 'ForecastId_Quantile')


# In[ ]:




test[test.Place=='US']
# In[ ]:


data_cc['US'].tail()


# In[ ]:


data_fs['US'].tail(10)


# In[ ]:


print(sub[309510:309510+(28+18)*2:2])


# In[ ]:


print(sub[309511:309511+(28+18)*2:2])


# In[ ]:


assert list(example_sub.columns) == list(sub.columns)
assert set(example_sub.ForecastId_Quantile) == set(sub.ForecastId_Quantile)


# In[ ]:


sub


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:





# ### Metric Calculation

# In[ ]:


cs = ['ConfirmedCases', 'Fatalities']


# In[ ]:


matched = pd.merge(test_wp[pdt + [c for c in test_wp if any(z in c for z in cs)]].dropna(),
             train_bk,
                    on = pdt)
matched.head()


# In[ ]:





# In[ ]:


matched_nq = pd.merge(test_nq[pdt + [c for c in test_nq if any(z in c for z in cs)]].dropna(),
             train_bk,
                    on = pdt)
# matched.head()


# In[ ]:


matched_rw = pd.merge(test_rw[pdt + [c for c in test_rw if any(z in c for z in cs)]].dropna(),
             train_bk,
                    on = pdt)
# matched.head()

matched_nqmatched_rw
# In[ ]:


def rw_quantile_loss(true, pred, quantile = 0.5):
    loss = np.where(true >= pred, 
                        quantile*(true-pred),
                        (1-quantile)*(pred - true) )
    return loss   
    

c = cs[0]
matched.Weight[matched.Target == c]\
                                                * rw_quantile_loss( matched.TargetValue[matched.Target == c], 
                                                        matched[c+"5_pred"][matched.Target == c], 0.05)matched.tail()
# In[ ]:


cs


# In[ ]:


matched_ws = []
for c in cs:
    matched_c = matched[matched.Target == c].copy()
    matched_c['error_5'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"5_pred"], 0.05)
    matched_c['error_50'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"_pred"], 0.50)
    matched_c['error_95'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"95_pred"], 0.95)
    matched_ws.append(matched_c)
matched_ws = pd.concat(matched_ws)
matched_ws['error'] = matched_ws.error_5 + matched_ws.error_50 + matched_ws.error_95


# In[ ]:


matched_nq_l = []
for c in cs:
    matched_c = matched_nq[matched_nq.Target == c].copy()
    matched_c['error_5'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"5_pred"], 0.05)
    matched_c['error_50'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"_pred"], 0.50)
    matched_c['error_95'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"95_pred"], 0.95)
    matched_nq_l.append(matched_c)
matched_nq = pd.concat(matched_nq_l)
matched_nq['error'] = matched_nq.error_5 + matched_nq.error_50 + matched_nq.error_95


# In[ ]:


matched_rw_l = []
for c in cs:
    matched_c = matched_rw[matched_rw.Target == c].copy()
    matched_c['error_5'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"5_pred"], 0.05)
    matched_c['error_50'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"_pred"], 0.50)
    matched_c['error_95'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"95_pred"], 0.95)
    matched_rw_l.append(matched_c)
matched_rw = pd.concat(matched_rw_l)
matched_rw['error'] = matched_rw.error_5 + matched_rw.error_50 + matched_rw.error_95

matched_rw
# In[ ]:





# In[ ]:


matched_ws.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:20]


# In[ ]:


matched_nq.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:10]


# In[ ]:


matched_rw.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:10]


# In[ ]:



(matched_nq.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:10]  / matched_rw.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:10] ) **2


# In[ ]:


(1.3)**2


# In[ ]:


def pickWeights(x, y):
    w = y**4 / (x**4 + y**4 )
    return w


# In[ ]:


nq_wt = pickWeights(matched_nq.groupby(['Place']).sum().iloc[:,-4:],
    matched_rw.groupby(['Place']).sum().iloc[:,-4:]);


# In[ ]:


nq_wt


# In[ ]:


nq_wt.loc['US']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


matched_blend = matched_nq.copy()
matched_blend.iloc[:, 3:5] = matched_nq.iloc[:, 3:5].multiply(
                                    matched_nq.Place.map(nq_wt.error_50), axis='index') \
                           + matched_rw.iloc[:, 3:5].multiply(
                                     ( 1 - matched_nq.Place.map(nq_wt.error_50)), axis='index') 
matched_blend.iloc[:, 5:7] = matched_nq.iloc[:, 5:7].multiply(
                                    matched_nq.Place.map(nq_wt.error_5), axis='index') \
                           + matched_rw.iloc[:, 5:7].multiply(
                                     ( 1 - matched_nq.Place.map(nq_wt.error_5)), axis='index') 
matched_blend.iloc[:, 7:9] = matched_nq.iloc[:, 7:9].multiply(
                                    matched_nq.Place.map(nq_wt.error_95), axis='index') \
                           + matched_rw.iloc[:, 7:9].multiply(
                                     ( 1 - matched_nq.Place.map(nq_wt.error_95)), axis='index') 


# In[ ]:


matched_blend


# In[ ]:


matched_final = []
for c in cs:
    matched_c = matched_blend[matched_blend.Target == c].copy()
    matched_c['error_5'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"5_pred"], 0.05)
    matched_c['error_50'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"_pred"], 0.50)
    matched_c['error_95'] = matched_c.Weight                                                * rw_quantile_loss( matched_c.TargetValue, 
                                                        matched_c[c+"95_pred"], 0.95)
    matched_final.append(matched_c)
matched_final = pd.concat(matched_final)
matched_final['error'] = matched_final.error_5 + matched_final.error_50 + matched_final.error_95

matched_nq.groupby(['Place']).sum().iloc[:,-4:]
# In[ ]:





# In[ ]:




nq_error = matched_ws.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:20]rw_error = matched_ws.groupby(['Place']).sum().iloc[:,-4:].sort_values('error', ascending=False)[:20]nq_error[:10]both = nq_error.merge(rw_error,right_index=True,left_index=True,
                  suffixes = ['_NQ', '_RW' ])
both[[ c for c in both if '5' not in c]]both
# In[ ]:


train.Date.max()


# In[ ]:


matched_rw.error.mean()/3


# In[ ]:


matched_nq.error.mean()/3


# In[ ]:



matched_final.error.mean()/3


# In[ ]:


nq_wt.drop(columns='error').plot(kind='hist', bins = 100);

nq_wt
# In[ ]:


nq_wt.sort_values('error')


# In[ ]:


nq_wt.to_csv('nq_wt.csv')


# In[ ]:


matched_final.to_csv('matched_final.csv')


# In[ ]:





# In[ ]:





# In[ ]:


matched_final


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


nq_wt


# In[ ]:





# In[ ]:


np.round(matched_final.groupby('Target')[[c for c in matched_ws.columns if 'error_' in c]].mean()/3, 3)


# In[ ]:





# In[ ]:


(matched_final.groupby('Place').error.sum().sort_values(ascending=False)[:50] / matched_final.error.sum())    .plot(kind='pie');


# In[ ]:


for place in matched_final.groupby('Place').error.sum().sort_values(ascending=False)[:20].index:
    matched_final[(matched_final.Place==place) & (matched_final.Target=='ConfirmedCases')]            .set_index('Date')            [['ConfirmedCases_pred',  'ConfirmedCases5_pred','ConfirmedCases95_pred' ,'TargetValue']]        .plot(title = '{} - Confirmed Cases'.format(place));
        


# In[ ]:


for place in matched_final.groupby('Place').error.sum().sort_values(ascending=False)[:20].index:
    matched_final[(matched_final.Place==place) & (matched_final.Target=='Fatalities')]            .set_index('Date')[['Fatalities_pred','Fatalities5_pred','Fatalities95_pred','TargetValue']]    .plot(title = '{} - Fatalities'.format(place));
        


# In[ ]:




