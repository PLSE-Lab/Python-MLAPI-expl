#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Daily Counts Model (Northquay)

# In[ ]:





# ### Run Settings

# In[ ]:


PRIVATE = True


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
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


pd.options.display.max_rows = 999


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


# path = '/kaggle/input/c19week3'
input_path = '/kaggle/input/covid19-global-forecasting-week-5'

# %% [code]
train = pd.read_csv(input_path + '/train.csv')
test = pd.read_csv(input_path  + '/test.csv')
sub = pd.read_csv(input_path + '/submission.csv')
example_sub = sub


# In[ ]:


print(train.Date.max())


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


# In[ ]:


if PRIVATE:
    test = test[ pd.to_datetime(test.Date) >  train.Date.max()]
    pp = 'private'
else:
    pp = 'public'

train = train[train.Date < test.Date.min()]


# ### Basics

# In[ ]:


pdt = ['Place', 'Date', 'Target']


# In[ ]:


train['Impact'] = train['TargetValue'] * train['Weight']


# In[ ]:


population = train.groupby('Place').Population.mean()
state = train.groupby('Place').Province_State.first()


# In[ ]:


train_avg = train.groupby(['Place', 'Date', 'Target']).mean()


# In[ ]:


train[(train.Target == 'Fatalities') & (train.Province_State.isnull())].groupby('Country').sum()    .sort_values('TargetValue', ascending=False)[:20]

train[(train.Target == 'Fatalities') & 
      (~train.Province_State.isnull()) & 
      (train.County.isnull())].groupby('Country').Impact.sum().sort_values()
# In[ ]:




train[train.Target == 'Fatalities'].groupby('Country').Impact.sum().sort_values(ascending=False)
# In[ ]:


train[train.Target == 'Fatalities'].groupby('Place').Impact.sum().sort_values(ascending=False)[:20]


# In[ ]:


train[train.Target == 'Fatalities'].groupby('Place').Weight.mean().sort_values().plot(kind='hist', bins = 250);

train[train.Target == 'Fatalities'].groupby('Place').Weight.mean().sort_values()train_avg[train_avg.index.get_level_values(2)=='ConfirmedCases'].sum()
# In[ ]:


top300 = train_avg[train_avg.index.get_level_values(2)=='ConfirmedCases'].reset_index()    .pivot('Date', 'Place', 'Impact').sum().sort_values(ascending = False)[:300]
top300[:30]


# In[ ]:





# In[ ]:


# normalize (down only) based on ewm_10 - ewm_40;
# multiply all scale features by norm factor, with noise 
# oversample regions with high norm factor


# In[ ]:


train_pivot_cc = train_avg[train_avg.index.get_level_values(2)=='ConfirmedCases'].reset_index()    .pivot('Date', 'Place', 'TargetValue')


# In[ ]:


train_pivot_f = train_avg[train_avg.index.get_level_values(2)=='Fatalities'].reset_index()    .pivot('Date', 'Place', 'TargetValue')


# In[ ]:


train_pivot_cc[top300.index].plot(legend = False);


# In[ ]:


train_pivot_cc.sum().sum()
train_pivot_cc.sum().sort_values(ascending=False)[:10]


# In[ ]:


# very strong argument US Cases;  New York


# In[ ]:


(train_pivot_cc.ewm(span=5).mean().iloc[-1,:] 
     / np.clip(train_pivot_cc.ewm(span=5).mean().iloc[-11,:], 10, None)).plot(kind='hist', bins = 250);


# In[ ]:





# In[ ]:


np.log((1 + train_pivot_cc.cumsum()) / population).max().plot(kind='hist', bins = 250);


# In[ ]:


np.exp(-7)
# any case prev below 0.001 gets  oversampled


# In[ ]:


(train_pivot_f.cumsum() / population).max().sort_values(ascending=False)[:20]


# In[ ]:


(train_pivot_cc.cumsum() / population).max().sort_values(ascending=False)[:20]


# In[ ]:


(population.sort_values(ascending=False) // 10e6)[:50]


# In[ ]:


(train_pivot_cc.cumsum() / population).max()[['Brazil', 'Russia', 'India']]


# In[ ]:





# In[ ]:


# DIVIDE BY ESTIMATED MAX OUTBREAK-- easy curve fit !!!
# should be a VERY EASY MODEL to get within an OOM or two;


# In[ ]:





# In[ ]:


(train_pivot_cc.ewm(span=5).mean().iloc[-1,:] 
     / np.clip(train_pivot_cc.ewm(span=5).mean().iloc[-16,:], 1000, None)).sort_values()[-10:]


# In[ ]:


# Brazil and Russia suging to levels only seen within the US -- very problematic on an absolute basis, vs. OS;
# rest are fine;


# In[ ]:


train_pivot_cc[top300.index].ewm(span=3).mean().plot(legend = False);


# ### Weighting and Resampling

# In[ ]:


WEIGHTS  = train[train.Target == 'Fatalities'].groupby('Place').Weight.mean().sort_values()
WEIGHTS[:20]


# In[ ]:


train[train.Target == 'Fatalities'].groupby('Place').Weight.mean().plot(kind='hist', bins = 250);


# In[ ]:


ULIM_CC = 1000
ULIM_FS = 100

ULIM_POP = 1e6


# In[ ]:


# downscale and upsample large places to keep model stationary
POP_SCALING = np.clip((population // ULIM_POP), 1, 30)
POP_SCALING.sort_values(ascending=False)[:300:10]


# In[ ]:




train_pivot_cc['US'].tail(20)
# In[ ]:


CC_SCALING =     pd.DataFrame([
        np.clip( train_pivot_cc.quantile(0.999999, axis = 0) // (ULIM_CC * 2), 1, 40),
        np.clip( train_pivot_cc.quantile(0.9, axis = 0) // (ULIM_CC), 1, 40)  ]).max()
CC_SCALING.sort_values(ascending=False)[:10]


# In[ ]:


train_pivot_f.max().sort_values()


# In[ ]:


FS_SCALING =     pd.DataFrame([
        np.clip( train_pivot_f.quantile(0.999999, axis = 0) // (ULIM_FS * 1), 1, 40),
        np.clip( train_pivot_f.quantile(0.9, axis = 0) // (ULIM_FS), 1, 40)  ]).max()
FS_SCALING.sort_values(ascending=False)[:10]


# In[ ]:


CC_SCALING = pd.DataFrame([POP_SCALING, CC_SCALING]).max()
CC_SCALING.sort_values(ascending=False)[:20]


# In[ ]:


FS_SCALING = pd.DataFrame([POP_SCALING, FS_SCALING]).max()
FS_SCALING.sort_values(ascending=False)[:20]


# In[ ]:


# shouldn't be downsampling Russia or Brazil at all;

train_pivot_f.sum().sum()
train_pivot_f.stack().sort_values(ascending=False)[:20]
# In[ ]:


600000*35 / 100


# In[ ]:


(train_pivot_cc / CC_SCALING).max().max()    / ((train_pivot_cc / CC_SCALING).sum().sum() / 30)


# In[ ]:


(train_pivot_cc / FS_SCALING).max().max()    / ((train_pivot_cc / FS_SCALING).sum().sum() / 30)


# In[ ]:




(train_pivot_cc / CC_SCALING).max().sort_values(ascending=False)[:20]train_pivot_cc['USMichigan']
# In[ ]:


(train_pivot_cc / CC_SCALING)[top300.index]    .ewm(span = 1).mean().plot(legend=False, linewidth = 1);


# In[ ]:


(train_pivot_cc / CC_SCALING)[top300.index]         .ewm(span = 3).mean().plot(legend=False, linewidth = 0.5);

train.groupby('Target').Impact.sum()train_pivot_f
# In[ ]:


train_pivot_f[top300.index].plot(legend = False);


# In[ ]:


train_pivot_f[top300.index][[c for c in top300.index if 'US' in c]].plot(legend = False);


# In[ ]:


(train_pivot_f['US'] - train_pivot_f['US_New York']).plot();


# In[ ]:





# In[ ]:


(train_pivot_f / FS_SCALING)[top300.index]        .ewm(span = 1).mean().plot(legend=False, linewidth = 1);


# In[ ]:


(train_pivot_f / FS_SCALING)[top300.index]        .ewm(span = 3).mean().plot(legend=False, linewidth = 0.5);


# In[ ]:





# In[ ]:


# The EWM x-overs need to be EWM RATIOS (EWM_10d / EWM_30d !!!)
# Load the model up with percentiles and medians (may matter more than means)
#

train_pivot_f
# In[ ]:


# USNew York and US need to be merged into USNew York in the fatality model--too much overlap in peak days;
# all other aggregations are small enough to be fine (Au states, etc.)


# In[ ]:


train_pivot_f[top300.index].ewm(span=10).mean().quantile(0.99, axis = 1).tail(10)


# In[ ]:


train_pivot_f[top300.index].plot(legend = False);


# In[ ]:


train_pivot_f[top300.index].ewm(span=10).mean().plot(legend = False);

train_pivot_f[top300.index].ewm(span=10).mean()\
     .iloc[-30:,:]
# In[ ]:





# ### Features

# In[ ]:





# In[ ]:





# In[ ]:


train_pivot_cc.shape


# In[ ]:




test.Date.min()
# In[ ]:





# ### Begin Basic Features

# In[ ]:


data_cc = train_pivot_cc
data_fs = train_pivot_f


# In[ ]:


def columize(pivot_df):
    return pivot_df.reset_index().melt(id_vars = 'Date', value_name = 'Value').Value


# In[ ]:


dataset = data_cc.astype(np.float32).reset_index().melt(id_vars='Date', value_name = 'ConfirmedCases')
dataset = dataset.merge(data_fs.astype(np.float32).reset_index()                            .melt(id_vars='Date', value_name = 'Fatalities'),
                        on = ['Date', 'Place'])


# In[ ]:





# ### Population Features

# In[ ]:


dataset['population'] = dataset.Place.map(np.log(population)) 


# In[ ]:


train['CountryState'] = (train.Country + '_' + train.Province_State).fillna('X');


# In[ ]:


state_population = train[train.County.isnull() & ~train.Province_State.isnull()].groupby('CountryState').Population.mean()


# In[ ]:


dataset['state_population'] =     np.log(dataset.Place.map(train.groupby('Place').first().CountryState).map(state_population) + 1).fillna(0)


# In[ ]:


dataset['place_type'] = dataset.Place.str.split('_').apply(len)


# In[ ]:


dataset['county_pct_state_population'] = np.where(dataset.place_type == 3,
                                                  dataset.population - dataset.state_population, 0)


# In[ ]:


max_county_population = train[~train.County.isnull()].groupby('CountryState').Population.max()


# In[ ]:


dataset['largest_county'] = np.where(dataset.place_type >= 2, 
                                     np.log( 1+ dataset.Place.map(train.groupby('Place').CountryState.first())\
                                            .map(max_county_population) ), -10 )
dataset.largest_county.fillna(-15, inplace=True)

dataset.groupby('Place').largest_county.first().sort_values().plot(kind='hist', bins = 250);
# In[ ]:


dataset['largest_county_vs_state_population'] =     dataset.largest_county - dataset.state_population
dataset.largest_county_vs_state_population.plot(kind='hist', bins = 250);

dataset.groupby('Place').largest_county_vs_state_population.first().sort_values()[::40]dataset.county_pct_state_population.plot(kind='hist', bins = 250);
# In[ ]:


PF_NOISE = 0.0;


# In[ ]:


dataset.population += np.random.normal(0, PF_NOISE, len(dataset))
dataset.state_population += np.random.normal(0, PF_NOISE, len(dataset))
dataset.county_pct_state_population += np.random.normal(0, 2 * PF_NOISE, len(dataset))
dataset.largest_county += np.random.normal(0, 2 * PF_NOISE, len(dataset))
dataset.largest_county_vs_state_population += np.random.normal(0, PF_NOISE, len(dataset))

dataset.population.plot(kind='hist', bins = 250);
dataset.state_population.plot(kind='hist', bins = 250);
dataset.county_pct_state_population .plot(kind='hist', bins = 250);


# In[ ]:


dataset.largest_county.plot(kind='hist', bins = 250);

dataset.largest_county_vs_state_population.plot(kind='hist', bins = 250);


# In[ ]:





# In[ ]:





# ### Various EDA

# In[ ]:


dataset[(dataset.Place.str.slice(0,2)=='US') & (dataset.ConfirmedCases >= 0)]


# In[ ]:


dataset[(dataset.Place.str.slice(0,2)=='US') & (dataset.ConfirmedCases > 0)].sort_values('Date')[::5]


# In[ ]:


data_cc['US'].plot();


# In[ ]:


dataset.dtypes

dataset.ConfirmedCases.max()
# In[ ]:


len(dataset)

data_cc['USNew York'].plot()
# ### Calc Features
data_cc['USMinnesota'].plot();
# In[ ]:


for window in [2, 4, 7, 14, 21, 35, 63]:
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
        

dataset[[c for c in dataset.columns if 'zero' in c]].plot(kind='hist', bins = 250);
# In[ ]:


dataset[dataset.Place=='USNew York'].set_index('Date').iloc[-100:,:]        [[c for c in dataset.columns if 'cc' in c and ('stdev_over' in c) 
                 ]]\
        .plot();


# In[ ]:


dataset[dataset.Place=='USNew York'].set_index('Date').iloc[-100:,:]        [[c for c in dataset.columns if 'cc' in c and ('low' in c or 'high' in c or 'median' in c) and
                 '14d' in c]]\
        .plot();


# In[ ]:


0.35 * 500 * 2 * 30 / 3 
# with 500 features, need to chop down to 33% of data (toss early empties) to get to 3.5gb data


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


# ### Per Capita Rates (Daily and Cumsum)
train_cc.shape
dataset.shape
# In[ ]:


# ensure indicies are aligned
assert (train_cc.Place != dataset.Place.values).sum() == 0
assert (train_fs.Place != dataset.Place.values).sum() == 0
assert (train_cc.Date != dataset.Date.values).sum() == 0
assert (train_fs.Date != dataset.Date.values).sum() == 0


# In[ ]:





# In[ ]:


len(dataset)


# In[ ]:


data_cc.shape


# In[ ]:





# In[ ]:


((data_cc).apply(np.sign) * (((data_cc).apply(np.abs) ).apply(np.log)                                        - (population + 1).apply(np.log) )).fillna(0).stack().shape


# In[ ]:


dataset = dataset.merge(pd.Series(((data_cc).apply(np.sign) * (((data_cc).apply(np.abs) ).apply(np.log)                                        - (population + 1).apply(np.log) )).fillna(0).stack()
                                  , name = 'ccdailypercapita'), on=['Place', 'Date'], copy=False)
dataset = dataset.merge(pd.Series(((data_fs).apply(np.sign) * (((data_fs).apply(np.abs) ).apply(np.log).fillna(0)                                        - (population + 1).apply(np.log) )).fillna(0).stack()
                                  , name = 'fsdailypercapita'), on=['Place', 'Date'], copy=False)


# In[ ]:


dataset = dataset.merge(pd.Series(((data_cc.cumsum() + 1).apply(np.log)                                        - (population + 1).apply(np.log) ).stack()
                                  , name = 'ccpercapita'), on=['Place', 'Date'], copy=False)
dataset = dataset.merge(pd.Series(((data_fs.cumsum() + 1).apply(np.log)                                        - (population + 1).apply(np.log) ).stack()
                                , name = 'fspercapita'), on=['Place', 'Date'], copy=False)


# In[ ]:


len(dataset)


# In[ ]:


dataset[[c for c in dataset.columns if 'percapita' in c and 'daily' not in c]].plot(kind='hist', bins = 250);


# In[ ]:


dataset[[c for c in dataset.columns if 'percapita' in c and 'daily' in c]].plot(kind='hist', bins = 250);


# In[ ]:


dataset.Date

datasetlen(dataset)dataset.Date
# In[ ]:


dataset['dayofweek'] = dataset.Date.dt.dayofweek


# In[ ]:


dataset.cckurtosis_7d.plot(kind='hist', bins = 250);

dataset.isnull().sum()
# In[ ]:


dataset.fillna(-10, inplace=True);
dataset.iloc[:, 4:] = dataset.iloc[:, 4:].astype(np.half, errors = 'ignore')

dataset
# In[ ]:





# In[ ]:


total_cc = data_cc.cumsum()
total_fs = data_fs.cumsum()


# In[ ]:


dataset.drop(columns = dataset.filter(like='days').columns, inplace=True)


# In[ ]:


firsts = [ (total_cc >= 1, 'days_since_first_case'),
            (total_cc >= 10, 'days_since_tenth_case'),
             (total_cc >= 100, 'days_since_hundredth_case'),
              (total_cc >= 1000, 'days_since_thousandth_case'),

          (total_fs >= 1, 'days_since_first_fatality'),
            (total_fs >= 10, 'days_since_tenth_fatality'),
             (total_fs >= 100, 'days_since_hundredth_fatality'),
         ]

for f in firsts:
    dataset = dataset.merge(pd.Series( f[0].idxmax().where((f[0].max() > 0), np.nan)                                               , name = f[1]), on='Place', copy=False)
    dataset[f[1]] = np.clip((dataset.Date - dataset[f[1]]).dt.days, -1, None).fillna(-1)

dataset[dataset.days_since_thousandth_fatality > -1]dataset[dataset.Place=='USNew York'].iloc[-20:, -10:]
# In[ ]:





# In[ ]:




Dataset Features:
#   EWMs of various windows
#    Include per capita rates on both;
#   Day of Week

#     Holiday (Easter, Memorial Day basically)
#    Stdev, Various Percentiles, Skewness, Kurtosis, Min, Max, etc.
# In[ ]:


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


def cfr(case, fatality):
    cfr_calc = np.log(    (fatality                                          + np.clip(0.015 * case, 0, 0.3))                             / ( case + 0.1) )
    return np.where(np.isnan(cfr_calc) | np.isinf(cfr_calc),
                           BLCFR, cfr_calc)


# In[ ]:


# %% [code]
BLCFR = dataset[(dataset.days_since_first_case == 0)].log_cfr.median()
dataset.log_cfr.fillna(BLCFR, inplace=True)
dataset.log_cfr = np.where(dataset.log_cfr.isnull() | np.isinf(dataset.log_cfr),
                           BLCFR, dataset.log_cfr)
BLCFR


# In[ ]:


dataset.log_cfr


# In[ ]:


# %% [code]  ** SLOPPY but fine
dataset['log_cfr_3d_ewm'] = BLCFR +                 (dataset.log_cfr - BLCFR).ewm(span = 3).mean()                       * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
                     
dataset['log_cfr_8d_ewm'] = BLCFR +                 (dataset.log_cfr - BLCFR).ewm(span = 8).mean()                       * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/8, 0, 1), 2)

dataset['log_cfr_20d_ewm'] = BLCFR +                 (dataset.log_cfr - BLCFR).ewm(span = 20).mean()                       * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/20, 0, 1), 2)

dataset['log_cfr_3d_20d_ewm_crossover'] = dataset.log_cfr_3d_ewm - dataset.log_cfr_20d_ewm


# %% [code]
dataset.drop(columns = 'log_cfr', inplace=True)

dataset.log_cfr_8d_ewm.plot(kind='hist', bins = 250);
# In[ ]:





# ### Per Capita vs. World and Similar Countries

# In[ ]:


date_total_cc = np.sum(total_cc, axis = 1)
date_total_fs = np.sum(total_fs, axis = 1)

populationdate_total_cc (total_cc + 1).apply(np.log)\
                                        - (population + 1).apply(np.log)\
                                   
# In[ ]:


# %% [code]
dataset['ConfirmedCases_percapita_vs_world'] = columize( ( (total_cc + 1).apply(np.log)                                        - (population + 1).apply(np.log))                                     .subtract( (date_total_cc + 1).apply(np.log), axis = 'index')                                        + np.log(population.sum() + 1) )
                                        

dataset.ConfirmedCases_percapita_vs_world.plot(kind='hist', bins = 250);
# In[ ]:


dataset['Fatalities_percapita_vs_world'] =  columize( ( (total_fs + 1).apply(np.log)                                        - (population + 1).apply(np.log))                                     .subtract( (date_total_fs + 1).apply(np.log), axis = 'index')                                        + np.log(population.sum() + 1) )
                                        


# In[ ]:


dataset['cfr_vs_world'] = dataset.log_cfr_3d_ewm                             -    np.log(    date_total_fs.loc[dataset.Date]                               /   date_total_cc.loc[dataset.Date]).values

dataset.cfr_vs_world.plot(kind='hist', bins = 250);
# In[ ]:





# ### Proximity: Growth Rates (EWM Crosses) and Implied Based on Per Capitas

# In[ ]:


# COMPARISONS:
# cumsum cc/fs per capita vs others
# X-day EWM per capita vs others

# vs. continent is nice, but unnecessary;

# vs. rest of country

# vs. other counties in this state (if it's a county)


# In[ ]:





# ### Compare to State
train_cctrain_cc[~train_cc.Province_State.isnull()].groupby(['Place']).Group.first()# compare per capitas with rest of state;
# calculate state-level per capitas
( (total_cc + 1).apply(np.log)\
                                        - (population + 1).apply(np.log))\
#                                      .subtract( (date_total_cc + 1).apply(np.log), axis = 'index')\
#                                         + np.log(population.sum() + 1) )total_cctrain_cc.set_index('County').groupby(['Country', 'Province_State'])total_ccstate
# #### Continent
cont_date_totals = dataset.groupby(['Date', 'continent_generosity']).sum()dataset['ConfirmedCases_percapita_vs_continent_mean'] = 0
dataset['Fatalities_percapita_vs_continent_mean'] = 0
dataset['ConfirmedCases_percapita_vs_continent_median'] = 0
dataset['Fatalities_percapita_vs_continent_median'] = 0

for cg in dataset.continent_generosity.unique():
    ps = dataset.groupby("Place").last()
    tp = ps[ps.continent_generosity==cg].TRUE_POPULATION.sum()
    print(tp / 1e9)
    for Date in dataset.Date.unique():
        cd =  dataset[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg)]\
                               [['ConfirmedCases', 'Fatalities', 'TRUE_POPULATION']]
#         print(cd)
        cmedian = np.median(np.log(cd.ConfirmedCases + 1)\
                                              - np.log(cd.TRUE_POPULATION+1))
        cmean = np.log(cd.ConfirmedCases.sum() + 1) - np.log(tp + 1)
        fmedian = np.median(np.log(cd.Fatalities + 1)\
                                              - np.log(cd.TRUE_POPULATION+1))
        fmean = np.log(cd.Fatalities.sum() + 1) - np.log(tp + 1)
        cfrmean = cfr( cd.ConfirmedCases.sum(),  cd.Fatalities.sum()   ) 
#         print(cmean)
        
#         break;
        
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'ConfirmedCases_percapita_vs_continent_mean'] = \
                                dataset['ConfirmedCases_percapita'] \
                                     - (cmean)
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'ConfirmedCases_percapita_vs_continent_median'] = \
                                dataset['ConfirmedCases_percapita'] \
                                     - (cmedian)
        
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'Fatalities_percapita_vs_continent_mean'] = \
                                dataset['Fatalities_percapita']\
                                    - (fmean)
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'Fatalities_percapita_vs_continent_median'] = \
                                dataset['Fatalities_percapita']\
                                    - (fmedian)
        
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'cfr_vs_continent'] = \
                                dataset.log_cfr_3d_ewm \
                            -    cfrmean

# #### Proximity Functions
all_places = dataset[['Place', 'latitude', 'longitude']].drop_duplicates().set_index('Place',
                                                                                    drop=True)
all_places.head()def surroundingPlaces(place, d = 10):
    dist = (all_places.latitude - all_places.loc[place].latitude)**2 \
                    + (all_places.longitude - all_places.loc[place].longitude) ** 2 
    return all_places[dist < d**2][1:n+1]

# surroundingPlaces('Afghanistan', 5)def nearestPlaces(place, n = 10):
    dist = (all_places.latitude - all_places.loc[place].latitude)**2 \
                    + (all_places.longitude - all_places.loc[place].longitude) ** 2
    ranked = np.argsort(dist) 
    return all_places.iloc[ranked][1:n+1]dgp = dataset.groupby('Place').last()
for n in [5, 10, 20]:   
    for place in dataset.Place.unique():
        nps = nearestPlaces(place, n)
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()
        dataset.loc[dataset.Place==place, 
                    'ratio_population_vs_nearest{}'.format(n)] = \
            np.log(dataset.loc[dataset.Place==place].TRUE_POPULATION.mean() + 1)\
                - np.log(tp+1)
         
        nbps =  dataset[(dataset.Place.isin(nps.index))]\
                            .groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

        nppc = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities + 1) - np.log(tp + 1))
        npp_cfr = cfr( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases,
                      nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities)
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_percapita_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].ConfirmedCases_percapita \
                            - nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_percapita_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].Fatalities_percapita \
                            - nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].log_cfr_3d_ewm \
                            - npp_cfr   
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_nearest{}_percapita'.format(n)] = nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_nearest{}_percapita'.format(n)] = nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_nearest{}'.format(n)] = npp_cfr
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_nearest{}_10d_slope'.format(n)] =   \
                               ( nppc.ewm(span = 1).mean() - nppc.ewm(span = 10).mean() ).values
        dataset.loc[
                (dataset.Place == place),
                    'Fatalities_nearest{}_10d_slope'.format(n)] =   \
                               ( nppf.ewm(span = 1).mean() - nppf.ewm(span = 10).mean() ).values
        
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_nearest{}_10d_slope'.format(n)] = \
                            ( npp_cfr_s.ewm(span = 1).mean()\
                                     - npp_cfr_s.ewm(span = 10).mean() ) .values
        dgp = dataset.groupby('Place').last()
for d in [5, 10, 20]:
    for place in dataset.Place.unique():
        nps = surroundingPlaces(place, d)
        dataset.loc[dataset.Place==place, 'num_surrounding_places_{}_degrees'.format(d)] = \
            len(nps)
        
        
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()
        
        dataset.loc[dataset.Place==place, 
                    'ratio_population_vs_surrounding_places_{}_degrees'.format(d)] = \
            np.log(dataset.loc[dataset.Place==place].TRUE_POPULATION.mean() + 1)\
                - np.log(tp+1)
        
        if len(nps)==0:
            continue;
            
        nbps =  dataset[(dataset.Place.isin(nps.index))]\
                            .groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

        nppc = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities + 1) - np.log(tp + 1))
        npp_cfr = cfr( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases,
                      nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities)
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_percapita_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].ConfirmedCases_percapita \
                            - nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_percapita_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].Fatalities_percapita \
                            - nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].log_cfr_3d_ewm \
                            - npp_cfr   
        
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_surrounding_places_{}_degrees_percapita'.format(d)] = nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_surrounding_places_{}_degrees_percapita'.format(d)] = nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_surrounding_places_{}_degrees'.format(d)] = npp_cfr
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_surrounding_places_{}_degrees_10d_slope'.format(d)] =   \
                               ( nppc.ewm(span = 1).mean() - nppc.ewm(span = 10).mean() ).values
        dataset.loc[
                (dataset.Place == place),
                    'Fatalities_surrounding_places_{}_degrees_10d_slope'.format(d)] =   \
                               ( nppf.ewm(span = 1).mean() - nppf.ewm(span = 10).mean() ).values
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_surrounding_places_{}_degrees_10d_slope'.format(d)] = \
                            ( npp_cfr_s.ewm(span = 1).mean()\
                                     - npp_cfr_s.ewm(span = 10).mean() ) .values
        for col in [c for c in dataset.columns if 'surrounding_places' in c and 'num_sur' not in c]:
    dataset[col] = dataset[col].fillna(0)
    n_col = 'num_surrounding_places_{}_degrees'.format(col.split('degrees')[0]\
                                                           .split('_')[-2])

    print(col)
    dataset[col + "_times_num_places"] = dataset[col] * np.sqrt(dataset[n_col])

dataset[dataset.Country=='US'][['Place', 'Date'] \
                                     + [c for c in dataset.columns if 'ratio_p' in c]]\
                [::50]# %% [code]
dataset.TRUE_POPULATION

# %% [code]
dataset.TRUE_POPULATION.sum()

# %% [code]
dataset.groupby('Date').sum().TRUE_POPULATION

# %% [code]
# ### Place-Specific Features II
dataset['first_case_ConfirmedCases_percapita'] = \
       np.log(dataset.first_case_ConfirmedCases + 1) \
          - np.log(dataset.TRUE_POPULATION + 1)

dataset['first_case_Fatalities_percapita'] = \
       np.log(dataset.first_case_Fatalities + 1) \
          - np.log(dataset.TRUE_POPULATION + 1)

dataset['first_fatality_Fatalities_percapita'] = \
       np.log(dataset.first_fatality_Fatalities + 1) \
          - np.log(dataset.TRUE_POPULATION + 1)

dataset['first_fatality_ConfirmedCases_percapita'] = \
        np.log(dataset.first_fatality_ConfirmedCases + 1)\
            - np.log(dataset.TRUE_POPULATION + 1)dataset['days_to_saturation_ConfirmedCases_4d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_4d_prior_slope         
dataset['days_to_saturation_ConfirmedCases_7d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_7d_prior_slope         

    
dataset['days_to_saturation_Fatalities_20d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_20d_prior_slope         
dataset['days_to_saturation_Fatalities_12d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_12d_prior_slope         
 dataset['days_to_3pct_ConfirmedCases_4d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 3.5) \
                            / dataset.ConfirmedCases_4d_prior_slope         
dataset['days_to_3pct_ConfirmedCases_7d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 3.5) \
                            / dataset.ConfirmedCases_7d_prior_slope         

    
dataset['days_to_0.3pct_Fatalities_20d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 5.8) \
                            / dataset.ConfirmedCases_20d_prior_slope         
dataset['days_to_0.3pct_Fatalities_12d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 5.8) \
                            / dataset.ConfirmedCases_12d_prior_slope         
 
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


# final day count percentiles
data_cc.iloc[-1, :].sort_values()[::data_cc.shape[1]//10]


# In[ ]:


# cumsum count percentiles
data_cc.cumsum().iloc[-1, :].sort_values()[::data_cc.shape[1]//10]


# In[ ]:


dataset.shape


# In[ ]:


dataset.iloc[:, 4:] = dataset.iloc[:, 4:].astype(np.half, errors = 'ignore')


# In[ ]:


.363*30*2*200 #4gb @ 200 features


# In[ ]:


dataset


# In[ ]:





# ### Build All Possible Intervals into Future

# In[ ]:


ramCheck()


# In[ ]:


len(dataset)


# In[ ]:


dataset = dataset[(dataset.Date >= datetime.datetime(2020, 3, 1) ) 
                      | (dataset.Place.str.slice(0,2) != 'US')]


# In[ ]:


len(dataset)


# In[ ]:


len(dataset.Place.unique())


# In[ ]:


dataset[(dataset.Place.str.slice(0,2)=='US') & (dataset.cc_ewm_63d > 0)].groupby('Place').first()    .sort_values('Date') [::300]


# In[ ]:


datas = []; data = None; model_data=None; g=gc.collect()


# In[ ]:


for window in range(1, 35):
    base = rollDates(dataset, window, True)
    datas.append(pd.merge(dataset[['Date', 'Place',
                 'ConfirmedCases', 'Fatalities']], base, on = ['Date', 'Place'],
                          how = 'right', 
            suffixes = ('_f', '')))
data = pd.concat(datas, axis =0, ignore_index=True); del datas


# In[ ]:


pd.to_datetime(data.Date.unique()[0])


# In[ ]:


daysofweek = {}
for date in data.Date.unique():
    daysofweek[date] = pd.to_datetime(date).weekday();


# In[ ]:


data['elapsed'] = (data.Date - data.Date_i).dt.days.astype(np.half)
data['finaldayofweek'] = data.Date.map(daysofweek)
data['dayofweek_diff'] = (data.dayofweek - data.finaldayofweek) % 7


# In[ ]:


g=gc.collect(); ramCheck()
# memCheck()

gc.get_stats()
# In[ ]:


test.Date.max()

# count percentiles
data.ConfirmedCases.sort_values()[::len(data)//10]# final day count percentiles
data[data.Date==train.Date.max()].ConfirmedCases.sort_values()[::len(data[data.Date==train.Date.max()])//10]
# final day count percentiles
np.mean(data.ConfirmedCases > 0)
np.mean(data[data.Date==train.Date.max()].ConfirmedCases > 0)np.mean(data.Fatalities > 0)
np.mean(data[data.Date==train.Date.max()].Fatalities > 0)np.clip(dataset[dataset.fs_ewm_63d >0 ].fs_ewm_14d, -1, 10).plot(kind='hist', bins = 250);np.clip(
    dataset[(dataset.fs_ewm_63d >0) & (dataset.Date == dataset.Date.max()) ]\
            .fs_ewm_14d, -1, 10).plot(kind='hist', bins = 250);np.mean(dataset.fs_ewm_63d == 0)
np.mean(dataset[dataset.Date == dataset.Date.max()].fs_ewm_63d == 0)np.mean(dataset.fs_ewm_63d >= 10)
np.mean(dataset[dataset.Date == dataset.Date.max()].fs_ewm_63d >= 10)
np.sum([c for c in dataset[ dataset.fs_ewm_63d >= 10 ].Fatalities]) \
    / np.sum([c for c in dataset[ dataset.fs_ewm_63d >= 0 ].Fatalities]) 
np.sum([c for c in dataset[ (dataset.Date == dataset.Date.max()) & (dataset.fs_ewm_63d >= 10)].Fatalities]) \
    / np.sum([c for c in dataset[ (dataset.Date == dataset.Date.max()) & (dataset.fs_ewm_63d >= 0)].Fatalities])# ~85% of the counts come from 2% of rows known in advance;
#  though variance could be WAY higher on smaller counts; AND eventually small counts COULD grow (?)

# fair to leave it all in one model; half zeros at end is very reasonable, lots of smalls, etc.
# gamma controls small buckets; child weight controls high ones; and num_leaves controls as well;
# one worry would be histograms don't capture feature granularity for big buckets (???)
# at 255, but most of the variance in top 3% (lightgbm may understand this)np.sum(dataset[dataset.Date==train.Date.max()].Fatalities )
np.sum([c for c in dataset[dataset.Date==train.Date.max()].Fatalities if c >= 10] )
np.sum([c for c in dataset[dataset.Date==train.Date.max()].Fatalities if c >= 100] )
np.sum([c for c in dataset[dataset.Date==train.Date.max()].Fatalities if c >= 500] )
np.clip(data[data.Date==train.Date.max()].Fatalities, -10, 25).plot(kind='hist', bins = 250);data.tail()
# In[ ]:





# In[ ]:


# decay away earlier series (early days at ~10-20% wt as mostly zeros, mostly irrelevant)

# likely VERY KEY TO USE GAMMA HERE -- don't want junk splits on small counts

# XX some argument for toss series where proximal is zero and population is very small (delete 90% right there)
# XX against upscale and downsample small series (but messy, 0->1 is a clear transition; 0->30 isn't)


# In[ ]:





# In[ ]:


# begin coding weighting scheme 
# recency decay
# downscale some series & their features and oversample them considerably


# In[ ]:




Actual Build:
    - EWM and mean crossovers and ratios, applied at bag time
    - Basic proximity features (intra-state, nearby Places, etc.)
    - Upscaling and downscaling and sampling code
    - Day of week moving averages etc.
    - Growth and decay multipliers, applied at bag time
    
    - View Errors--consider model just for largest regions only
    - Weighting--likely weight more recent data much more heavily
    - Drops?  Likely could drop more than half the data--if there is no outbreak yet within any proximity
    - Could drop most small series to accelerate training
    - Setup and view percentiles; make sure seem reasonable etc.
    
    - Code actual metric, view public test, compare with others and hand-draw etc.
    - Try to make a NY etc. curve myself in Excel and compare metrics
    - Experiment with incremental model, predict 1-20 days, then step again etc.
    - Plan for final day timing etc.
    - Consider LSTM; compare metrics given using no external features etc.
# In[ ]:




Per capita state rate 
Per capita state rate * county population 
Per capita state rate * county population - county rate [and this * days elapsed]
County rate / Per capita state rate * county population 

Per capita state daily rate
Per capita state daily rate * county population
Per capita state daily rate * county population - county daily rate [ and this * days elapsed ] 
County daily rate / Per capita state daily rate * county population 




NP.HALF does not work beyond 65k; cannot work for counts--need to use .iloc for any np.half changes
also cannot work for any cumsum variables--and thus any cumsum EWMs etc.
 - do .MAX() on every columns and  also check for smallest nonzero value; x[x>0].min()

should be safe for daily counts (never exceed 3k; fine);

----

ideal function would be start with current count EWM (7d), apply growth/decay rate that seems applicable from EWM crosses, and let the rate fade  to 0 over time with exponential decay e.g the 0.90 0.95 etc.
so maybe growing 10% per day and coming down to 3% growth
or maybe decaying 3% per day but eventually heads to 1% decay
etc.

AND THEN OVERLAY IT WITH A ROUGH DAILY RATIO --multiply counts by ewm_Xd to get a constant, and then get an EWM of past dailies in here; or median of last 4 days etc.

e.g. DAY / EWM(3) is a rough measure of typical day; and can put in various EWMs of these;
   or median of last 4 is a great measure as well; so is mean; and so is MAX and MIN for pctile finding

----

day of weeks are universal: weekend effect on fatalities; ideally just WEEKEND * PREDICTION
  (for those magical decay splines etc.) goes a long way;


going to build all feature combos INSIDE A FUNCTION that is run at bag stage;

---


should be easy to add back some state-level corrections and sup and containment VERY LATE (and check metric)
county latitude and longitude also becomes very easy, ~2 hours to find and add; then measure;


I highly trust the two-week CV this time; should be very accurate given the decays 
( and inaccurate on the growers that need to be benchmarked intuitively)



should be able to go back as deep as 3-4 weeks and still use that as CV as well;

pay lots of attention to errors and engineer to fix them, cut that by 10-20% in final sprint








 
    








TRUE_AGG function works here, except apply EXP() and sum it up;

also apply a standard exponential decay, EXP(-0.05x), i.e. counts fall by 1, 2%, 5% per day etc.
   off of a prior EWM;

smart to make some effort to DETECT CURRENT DECAY RATE based on 7d/ 14d /21d ewms-- and include it;

should have 2-10 features I'd view as beluga-quality; and can be picked based on one other feature etc.



# In[ ]:




# %% [code]
data['CaseChgRate'] = (np.log(data.ConfirmedCases_f + 1) - np.log(data.ConfirmedCases + 1))\
                            / data.elapsed;
data['FatalityChgRate'] = (np.log(data.Fatalities_f + 1) - np.log(data.Fatalities + 1))\
                            / data.elapsed;data.elapsed

falloff_hash = {}def true_agg(rate_i, elapsed, bend_rate):
    elapsed = int(elapsed)

    if (bend_rate, elapsed) not in falloff_hash:
        falloff_hash[(bend_rate, elapsed)] = \
            np.sum( [  np.power(bend_rate, e) for e in range(1, elapsed+1)] )
    return falloff_hash[(bend_rate, elapsed)] * rate_i
     true_agg(0.3, 30, 0.9)slope_cols = [c for c in data.columns if 
                      any(z in c for z in ['prior_slope', 'chg', 'rate'])
           and not any(z in c for z in ['bend', 'prior_slope_chg', 'Country', 'ewm', 
                                        ]) ] # ** bid change; since rate too stationary
print(slope_cols)

bend_rates = [1, 0.95, 0.90]
for bend_rate in bend_rates:
    bend_agg = data[['elapsed']].apply(lambda x: true_agg(1, *x, bend_rate), axis=1)
     
    for sc in slope_cols:
        if bend_rate < 1:
            data[sc+"_slope_bend_{}".format(bend_rate)] =  data[sc]  \
                                    * np.power((bend_rate + 1)/2, data.elapsed)
         
            data[sc+"_true_slope_bend_{}".format(bend_rate)] = \
                          bend_agg *  data[sc] / data.elapsed
            
        data[sc+"_agg_bend_{}".format(bend_rate)] =  data[sc] * data.elapsed \
                                * np.power((bend_rate + 1)/2, data.elapsed)
         
        data[sc+"_true_agg_bend_{}".format(bend_rate)] = \
                        bend_agg *  data[sc]
# ### Times Elapsed
slope_cols[:5]
for col in [c for c in data.columns if any(z in c for z in 
                               ['vs_continent', 'nearest', 'vs_world', 'surrounding_places'])]:
    data[col + '_times_days'] = data[col] * data.elapsed
# ### Saturation
data['saturation_slope_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1)) \
                                                    / data.elapsed
data['saturation_slope_Fatalities'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1)) \
                                                    / data.elapsed

data['dist_to_ConfirmedCases_saturation_times_days'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1)) \
                                                    * data.elapsed
data['dist_to_Fatalities_saturation_times_days'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1)) \
                                                    * data.elapsed
        


data['slope_to_1pct_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1) - 4.6) \
                                                    / data.elapsed
data['slope_to_0.1pct_Fatalities'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1) - 6.9) \
                                                    / data.elapsed

data['dist_to_1pct_ConfirmedCases_times_days'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1) - 4.6) \
                                                    * data.elapsed
data['dist_to_0.1pct_Fatalities_times_days'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1) - 6.9) \
                                                    * data.elapseddata['trendline_per_capita_ConfirmedCases_4d_slope'] = ( np.log(data.ConfirmedCases + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_4d_prior_slope * data.elapsed)
data['trendline_per_capita_ConfirmedCases_7d_slope'] = ( np.log(data.ConfirmedCases + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_7d_prior_slope * data.elapsed)
 

data['trendline_per_capita_Fatalities_12d_slope'] = ( np.log(data.Fatalities + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_12d_prior_slope * data.elapsed)
data['trendline_per_capita_Fatalities_20d_slope'] = ( np.log(data.Fatalities + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_20d_prior_slope * data.elapsed)

 data.groupby('Place').last()

 
def logHist(x, b = 150):
    returndata['log_fatalities'] = np.log(data.Fatalities + 1) #  + 0.4 * np.random.normal(0, 1, len(data))
data['log_cases'] = np.log(data.ConfirmedCases + 1) # + 0.2 *np.random.normal(0, 1, len(data))
data['is_China'] = (data.Country=='China') & (~data.Place.isin(['Hong Kong', 'Macau']))
# #### Noisy the EWMs
for col in [c for c in data.columns if 'd_ewm' in c]:
    data[col] += np.random.normal(0, 1, len(data)) * np.std(data[col]) * 0.2
    
# #### Final Features
data['is_province'] = 1.0* (~data.Province_State.isnull() )

data['log_elapsed'] = np.log(data.elapsed + 1)data.columns

data.columns[::19]

data.shape

logHist(data.ConfirmedCases)
# ### Final Data Cleanup

# In[ ]:




# # %% [code]
# data.drop(columns = ['TRUE_POPULATION'], inplace=True)

# %% [code]
data['final_day_of_week'] = data.Date_f.apply(datetime.datetime.weekday).astype(np.half)

# %% [code]
data['base_date_day_of_week'] = data.Date.apply(datetime.datetime.weekday)

# %% [code]
data['date_difference_modulo_7_days'] = (data.Date_f - data.Date).dt.days % 7# Clip Days-To
for c in data.columns.to_list():
    if 'days_to' in c:
        data[c] = data[c].where(~np.isinf(data[c]), 1e3)
        data[c] = np.clip(data[c], 0, 365)
        data[c] = np.sqrt(data[c])# Flag New Places              
new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &
      (train.ConfirmedCases == 0)
     ].Place

        
        
        # %% [code]


# %% [markdown]
# In[ ]:





# # II. Modeling

#  ### Data Prep
memCheck()
# In[ ]:


gc.collect()


# In[ ]:


ramCheck()


# In[ ]:





# In[ ]:


# Private vs. Public Training

if PRIVATE:
    data_test = data[ (data.Date_i == train.Date.max() ) & 
                     (data.Date.isin(test.Date.unique() ) ) ].copy()
else:
    MAX_PUBLIC = datetime.datetime(2020, 10, 11)
    data_test = data[ (data.Date_i == test.Date.min() - datetime.timedelta(1) ) & 
                     (data.Date.isin(test.Date.unique() ) ) &
                      (data.Date <= MAX_PUBLIC) ].copy()


# In[ ]:


ramCheck()

test.Date.min()
# In[ ]:


# %% [code]
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
[c for c in model_data.Place if '_' in c]model_data.tail()
# In[ ]:


y_cases = model_data.ConfirmedCases_f
y_fatalities = model_data.Fatalities_f
y_cfr = np.log(    (model_data.Fatalities_f                                          + np.clip(0.015 * model_data.ConfirmedCases_f, 0, 0.3))                             / ( model_data.ConfirmedCases_f + 0.1) )

places = model_data.Place


# In[ ]:


group_dict = {}
for place in dataset.Place.unique():
    group_dict[place] = '_'.join(place.split('_')[0:2])
group_dict['US'] = 'US_New York'


# In[ ]:


groups = model_data.Place.map(group_dict)


# In[ ]:


CC_SCALING.sort_values(ascending=False)[:20]

group_dictmodel_data.tail()
# In[ ]:


# model_data = model_data[~( 
#                             ( np.random.rand(len(model_data)) < 0.8 )  &
#                           ( model_data.Country == 'China') &
#                               (model_data.Date < datetime.datetime(2020, 2, 15)) )]

# %% [code]
x_dates = model_data[['Date_i', 'Date', 'Place']]

# x_dates.rename({'Date': 'Date_f'}, inplace=True)

# %% [code]


# In[ ]:


x = model_data.iloc[:, 6:].copy().drop(columns = ['Date_i'])
del model_data


# In[ ]:


g = gc.collect();
ramCheck()

data_test.Date.unique()

# %% [code]
test.Date.unique()
# In[ ]:


x_test =  data_test[x.columns].copy()


# In[ ]:


train.Date.max()
test.Date.max()

x.tail()
# ### Model Setup

# In[ ]:


# %% [code]
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, PredefinedSplit
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


# 
sklearn.metrics.make_scorer(score_func, 
                            greater_is_better=True, 
                            needs_proba=False, 
                            needs_threshold=False, **kwargs)SEED = 3np.random.seed(SEED)enet_params = { 'alpha': [   3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,  ],
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
                'max_depth': [  10, 12, 14, 16],
                'n_estimators': [ 50, 100, 150, 225, 350 ],   # continuous
                'min_split_gain': [0, 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1],
                'min_child_samples': [ 1, 2, 4, 7, 10, 14, 20, 30, 40, 70, 100, 140],
                'min_child_weight': [0], #, 1e-3],
                'num_leaves': [  20, 30, 50, 100],
                'learning_rate': [0.05, 0.07, 0.1],   #, 0.1],       
                'colsample_bytree': [0.1, 0.2, 0.33, 0.5, 0.65, 0.8, 0.9], 
                'colsample_bynode':[0.1, 0.2, 0.33, 0.5, 0.65, 0.81],
                'reg_lambda': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100,    ],
                'reg_alpha': [1e-5,  1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1, 1, 1, 10,  ], # 1, 10, 100, 1000, 10000],
                'subsample': [   0.9, 1],
                'subsample_freq': [1],
                'max_bin': [  50, 90, 125, 175, 255],
               }    


# In[ ]:


lgb_quantile_params = {
                'max_depth': [7, 10, 12, 14, 16],
                'n_estimators': [ 100, 150, 225, ],   # continuous
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
                'max_bin': [  50, 90, 125, 175, 255],
               }    


# In[ ]:


MSE = 'neg_mean_squared_error'
MAE = 'neg_mean_absolute_error'

def trainENet(x, y, groups, cv = 0, **kwargs):
    return trainModel(x, y, groups, 
                      clf = ElasticNet(normalize = True, selection = 'random', 
                                       max_iter = 3000),
                      params = enet_params, 
                      cv = cv, **kwargs)def trainETR(x, y, groups, cv = 0, n_jobs = 5,  **kwargs):
    clf = ExtraTreesRegressor(n_jobs = 1)
    params = et_params
    return trainModel(x, y, groups, clf, params, cv, n_jobs, **kwargs)
# In[ ]:


def trainLGB(x, y, groups, cv = 0, n_jobs = -1, **kwargs):
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  objective = 'mae'
                      )
    print('created lgb regressor')
    params = lgb_params
    
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
        if np.random.rand() < 1: # all shuffle, don't want overfit models, just reasonable
            folds = GroupShuffleSplit(n_splits = 5, 
                                 test_size= 0.2 + 0.10 * np.random.rand())
        else:
            folds = GroupKFold(4)
        print('running randomized search')
        clf = RandomizedSearchCV(clf, params, 
                            cv=  folds, 
                                 n_iter = 7, 
                                verbose = 1, n_jobs = n_jobs, scoring = cv)
        f = clf.fit(x, y, groups)
        print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  
        
        best = clf.best_estimator_;  print(best)
        print("Best Score: {}".format(np.round(clf.best_score_,4)))
        
        return best

def getSparseColumns(x, verbose = 0):
    sc = []
    for c in x.columns.to_list():
        u = len(x[c].unique())
        if u > 10 and u < 0.01*len(x) :
            sc.append(c)
            if verbose > 0:
                print("{}: {}".format(c, u))

    return scdef noisify(x, noise = 0.1):
    x = x.copy()
   # cols = x.columns.to_list()
    cols = getSparseColumns(x)
    for c in cols:
        u = len(x[c].unique())
        if u > 50:
            x[c].values[:] = x[c].values + np.random.normal(0, noise, len(x)) * np.std(x[c])
    return x;CC_SCALING
FS_SCALING
# In[ ]:


LT_DECAY_MAX = 0.3
LT_DECAY_MIN = -0.4


# In[ ]:





# In[ ]:


g = gc.collect()


# In[ ]:


ramCheck()


# In[ ]:


CC_SCALING

memCheck()np.exp(-1)
# In[ ]:


gc.collect()

np.exp(-1)x.tail().drop(columns = 
                    [c for c in data.columns if any(z in c for z in ['fs_', 'cc_'])]).iloc[:, 54:]
# In[ ]:


RESAMPLE_COLS = [c for c in x.columns if any(z in c for z in ['fs_', 'cc_'])]

x.loc[:, RESAMPLE_COLS]resample = CC_SCALINGx.loc[:, RESAMPLE_COLS] / resample
# In[ ]:


y_fatalities[x.elapsed == 10].sum()
y_fatalities[x.elapsed == 10].sort_values(ascending=False)[0:200].sum()


# In[ ]:


len(x)


# In[ ]:




np.exp(1)
# In[ ]:





# ### Explore x
np.clip(x.fs_ewm_63d, -1, 1).plot(kind='hist', bins = 250);(x[(x_dates.Date_i==x_dates.Date_i.max()) & (x.cc_ewm_63d == 0)])
# In[ ]:


final_row = x_dates.Date==x_dates.Date.max()


# In[ ]:


final_row.sum()
final_row.sum() / 255


# In[ ]:


(x[final_row].cc_ewm_21d == 0).sum()

(x[final_row].cc_ewm_63d == 0).sum()


# In[ ]:


final_li = (final_row & (x.elapsed == 30))
final_li.sum()
y_cases[final_li].sum()
# y_cases[final_li].sort_values(ascending=False) [ :: 300]
(y_cases[final_li]/places[final_li].map(CC_SCALING)).sort_values(ascending=False) [ ::300]


# In[ ]:


x[final_li].cc_ewm_63d.sort_values(ascending=False)[::300]


# In[ ]:


np.clip(  x[final_row]
        .cc_ewm_21d, -1, 4).plot(kind='hist', bins = 250);


# In[ ]:


np.clip(  y_cases[final_li
                  & (x.cc_ewm_63d < 1/10) & (x.elapsed > 0)]
        , -1, 30).plot(kind='hist', bins = 250);

x[x.cc_ewm_63d == 0][::10].groupby(x_dates[x.cc_ewm_63d == 0].Date_i[::10]).count()
# ### Downsampling

# In[ ]:


LOW_COUNT_SCALING = np.clip( x.cc_ewm_63d, 1/100, 1)
LOW_COUNT_SCALING_TEST = np.clip( x_test.cc_ewm_63d, 1/100, 1)

data_cc['US_New York'].plot()np.clip(x.cc_ewm_63d, -10, 100).plot(kind='hist', bins = 250);
# In[ ]:


x[final_li].cc_ewm_63d.sort_values()[::300]


# In[ ]:


x[final_li][np.random.rand(len(x[final_li])) < LOW_COUNT_SCALING[final_li] ].cc_ewm_63d.sort_values()[::100]


# In[ ]:


np.clip(y_cases[final_li]    , -5,200).plot(kind='hist', bins = 250);


# In[ ]:


len(y_cases[final_li][np.random.rand(len(x[final_li])) < LOW_COUNT_SCALING[final_li] ] )


# In[ ]:


np.clip(y_cases[final_li][np.random.rand(len(x[final_li])) < LOW_COUNT_SCALING[final_li] ]    , -5,200).plot(kind='hist', bins = 250);


# ### Now Run Models

# In[ ]:


def rescale_x(x, resample):
    x = x.astype(np.float32)
    for col in RESAMPLE_COLS:
        x[col] = x[col] / resample
    return x


# In[ ]:


SET_FRAC = 0.12


# In[ ]:


BAGS = 1


# In[ ]:


SIZE_RANGE = 0


# In[ ]:


SCALE_RANGE = 5


# In[ ]:


def runBags(x, y, groups, cv, bags = 3, model_type = trainLGB, 
            noise = 0.1, splits = None, weights = None, resample = None, **kwargs):
    models = []
    for bag in range(bags):
        print("\nBAG {}".format(bag+1))

        # set size picked randomly
        ssr =  SET_FRAC * np.exp(  - SIZE_RANGE * np.random.rand()  )
        
        # now decay for dates and weights
        date_falloff = 2 * ( 1/30 + 1/30 * np.random.rand()  ) # fastest ~15; slowest 30;
        if weights is not None:
            ssr = ssr * np.exp(-weights * date_falloff) * places.map(WEIGHTS)
            
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
#         print((yss.sort_values(ascending = False))[::len(xss)//10])
        
#         print("{:.1%} with no case history".format( (xss.cc_ewm_63d == 0).sum() / len(xss) ))
#         return []
# #         if DROPS:
# #             # drop 0-70% of the bend/slope/prior features, just for speed and model diversity
# #             for col in [c for c in x.columns if any(z in c for z in ['bend', 'slope', 'prior'])]:
# #                 if np.random.rand() < np.sqrt(np.random.rand()) * 0.7:
#                     x[col].values[:] = 0
            
        # 00% of the time drop all 'rate_since' features 
#         if np.random.rand() < 0.00:
#             print('dropping rate_since features')
#             for col in [c for c in x.columns if 'rate_since' in c]:    
#                 x[col].values[:] = 0
        
        # 20% of the time drop all 'world' features 
#         if np.random.rand() < 0.00:
#             print('dropping world features')
#             for col in [c for c in x.columns if 'world' in c]:    
#                 x[col].values[:] = 0
        
#         # % of the time drop all 'nearest' features 
#         if DROPS and (np.random.rand() < 0.30):
#             print('dropping nearest features')
#             for col in [c for c in x.columns if 'nearest' in c]:    
#                 x[col].values[:] = 0
        
#         #  % of the time drop all 'surrounding_places' features 
#         if DROPS and (np.random.rand() < 0.25):
#             print('dropping \'surrounding places\' features')
#             for col in [c for c in x.columns if 'surrounding_places' in c]:    
#                 x[col].values[:] = 0
        
        
        # 20% of the time drop all 'continent' features 
#         if np.random.rand() < 0.20:
#             print('dropping continent features')
#             for col in [c for c in x.columns if 'continent' in c]:    
#                 x[col].values[:] = 0
        
        # drop 0-50% of all features
#         if DROPS:
#         col_drop_frac = np.sqrt(np.random.rand()) * 0.5
#         for col in [c for c in x.columns if 'elapsed' not in c ]:
#             if np.random.rand() < col_drop_frac:
#                 x[col].values[:] = 0

        
#         x = noisify(x, noise)
        
        
#         if DROPS and (np.random.rand() < SUP_DROP):
#             print("Dropping supplemental country data")
#             for col in x[[c for c in x.columns if c in sup_data.columns]]:  
#                 x[col].values[:] = 0
                
#         if DROPS and (np.random.rand() < ACTIONS_DROP): 
#             for col in x[[c for c in x.columns if c in contain_data.columns]]:  
#                 x[col].values[:] = 0
# #             print(x.StringencyIndex_20d_ewm[::157])
#         else:
#             print("*using containment data")
            
#         if np.random.rand() < 0.6: 
#             x.S_data_days = 0
            
#       p1 =x.elapsed[ss].plot(kind='hist', bins = int(x.elapsed.max() - x.elapsed.min() + 1))
#         p1 = plt.figure();
#         break
#        print(Counter(groups[ss]))
#         print((ss).sum())
        models.append(model_type(xss, yss, groups[ss], cv, **kwargs))
    return models

# %% [code]


# In[ ]:


BAG_MULT = 1

x.shape


# In[ ]:



date_weights =  (1 * np.abs((x_dates.Date_i - test.Date.min()).dt.days) + 
                    2 * np.abs((x_dates.Date - test.Date.min()).dt.days))/3
                        


# In[ ]:





# In[ ]:


SINGLE_MODEL = False


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


# #### Cases

# In[ ]:


lgb_c_clfs = []; lgb_c_noise = []


# In[ ]:


for iteration in range(0, int(math.ceil(1 * BAGS))):
    for noise in [ 0.05, 0.1, 0.2, 0.3, 0.4  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        cv_group = groups
        
        lgb_c_clfs.extend(runBags(x, y_cases, 
                          cv_group, #groups
                          MAE, num_bags, trainLGB, verbose = 0, 
                                          noise = noise, weights = date_weights,
                                          resample = places.map(CC_SCALING) *  LOW_COUNT_SCALING

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
        cv_group = groups
        
        lgb_c5_clfs.extend(runBags(x, y_cases, 
                          cv_group, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise, weights = date_weights,
                                          resample = places.map(CC_SCALING) *  LOW_COUNT_SCALING,
                                   alpha = alpha

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
        cv_group = groups
        
        lgb_c95_clfs.extend(runBags(x, y_cases, 
                          cv_group, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise, weights = date_weights,
                                          resample = places.map(CC_SCALING) *  LOW_COUNT_SCALING,
                                   alpha = alpha

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
        cv_group = groups
        
        lgb_f5_clfs.extend(runBags(x, y_fatalities, 
                          cv_group, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise, weights = date_weights,
                                          resample = places.map(FS_SCALING) *  LOW_COUNT_SCALING,
                                   alpha = alpha

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
        cv_group = groups
        
        lgb_f95_clfs.extend(runBags(x, y_fatalities, 
                          cv_group, #groups
                          quantile_scorer(alpha), num_bags, trainLGBquantile, verbose = 0, 
                                          noise = noise, weights = date_weights,
                                          resample = places.map(FS_SCALING) *  LOW_COUNT_SCALING,
                                   alpha = alpha

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
#         print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * int(np.ceil(np.sqrt(BAG_MULT)))
#         if np.random.rand() < PLACE_FRACTION  :
#             cv_group = places
#             print("CV by Place")
#         else:
        cv_group = groups
#             print("CV by Country")
            
   
        lgb_f_clfs.extend(runBags(x, y_fatalities, 
                                  cv_group, #places, # groups, 
                                  MAE, num_bags, trainLGB, 
                                  verbose = 0, noise = noise,
                                  weights = date_weights,
                                  resample = places.map(FS_SCALING) * LOW_COUNT_SCALING

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
                                          weights = date_weights

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


f = avg_FI([lgb_c_clfs], x.columns, 25)

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

# %% [code]
# for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
#             'world', 'continent', 'nearest', 'surrounding']:
#     print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# %% [code]
# print("{}: {:.2f}".format('sup_data', 
#                        f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
# print("{}: {:.2f}".format('contain_data', 
#                    f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))


# In[ ]:


f = avg_FI([lgb_c95_clfs], x.columns, 25)


# In[ ]:


f = avg_FI([lgb_f95_clfs], x.columns, 25)


# In[ ]:




for clf_set in [lgb_c95_clfs, lgb_f95_clfs]:
    f = avg_FI([clf_set], x.columns, 25)
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
    return np.where( np.min(x,axis=0) > 0.1, gmean(np.clip(x, 0.1, None), axis = 0), np.median(x,axis = 0))
                    
#                     np.median(x,axis=0)
#     return (np.mean(x, axis=0) + np.median(x, axis=0))/2


# In[ ]:


def noisify(x, noise):
    return x


# In[ ]:


test_cc_scaling = data_test.Place.map(CC_SCALING) * LOW_COUNT_SCALING_TEST
test_fs_scaling = data_test.Place.map(FS_SCALING) * LOW_COUNT_SCALING_TEST

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


# In[ ]:


# %% [code]
pred = pd.DataFrame(np.hstack((np.transpose(c_preds),
                              np.transpose(f_preds))), index=x_test.index)
pred['Place'] = data_test.Place

pred['Date'] = data_test.Date_i
pred['Date_f'] = data_test.Date


# In[ ]:


# %% [code]
pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][30: 60]


# In[ ]:


# %% [code]
np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())], 2)[190:220:]


# In[ ]:


data_cc['US_New York'].tail()


# In[ ]:





# In[ ]:


train[train.County=='New York'].Population.mean()


# In[ ]:


pred.set_index('Place')[0].groupby('Place').max().sort_values()


# In[ ]:


# %% [code] {"scrolled":false}
np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][220:-20],2)

# %% [code]
c_preds.shape
x_test.shape
# In[ ]:





# ### Predict on Test Set
data_wp
# In[ ]:


### %% [code]
data_wp = data_test.copy()

data_wp['ConfirmedCases_pred'] = y_cases_pred_blended_full 
data_wp['Fatalities_pred'] = y_fatalities_pred_blended_full 
data_wp['ConfirmedCases5_pred'] = y_cases5_pred_blended_full 
data_wp['Fatalities5_pred'] = y_fatalities5_pred_blended_full 
data_wp['ConfirmedCases95_pred'] = y_cases95_pred_blended_full 
data_wp['Fatalities95_pred'] = y_fatalities95_pred_blended_full 

# data_wp['cfr_pred'] = y_cfr_pred_blended_full

len(data_test)
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


# In[ ]:


data_wp


# In[ ]:


base_date

# %% [code]
data_wp_ss = data_wp[data_wp.Date_i == base_date]
data_wp_ss = data_wp_ss.drop(columns='Date_i').rename(columns = {'Date_f': 'Date'})

# %% [code]
test_wp = pd.merge(test, data_wp_ss[['Date', 'Place', 'ConfirmedCases_pred', 'Fatalities_pred',
                                      'ConfirmedCases5_pred', 'Fatalities5_pred',
                                      'ConfirmedCases95_pred', 'Fatalities95_pred',
                                    'elapsed']], 
            how='left', on = ['Date', 'Place'])


# In[ ]:


test_wp.to_csv('test_wp.csv', index=False)


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

test_wp[pdt + [c for c in test_wp if any(z in c for z in cs)]].dropna().head()
# In[ ]:


matched = pd.merge(test_wp[pdt + [c for c in test_wp if any(z in c for z in cs)]].dropna(),
             train_bk,
                    on = pdt)
matched.head()


# In[ ]:





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

matched_ws
# In[ ]:


matched_ws.set_index(['Place','Date']).iloc[:,-6:]


# In[ ]:


matched_ws.error.mean()/3

matched_ws.error.sum()
# In[ ]:


pd.Series(matched_ws.set_index(['Place','Date']).iloc[:,-6:]    .groupby('Place').sum().error.sort_values(ascending=False)[: 50] / matched_ws.error.sum())            .plot(kind='pie');

matched_ws.set_index(['Place','Date']).iloc[:,-6:]\
    .groupby('Place').sum().sort_values('error', ascending=False)[:10]
# In[ ]:


matched_ws.set_index(['Place','Date']).iloc[:,-6:].sort_values('error', ascending=False)[:10]


# In[ ]:





# In[ ]:


np.round(matched_ws.groupby('Target')[[c for c in matched_ws.columns if 'error_' in c]].mean()/3, 3)


# In[ ]:


# techniques -- figure out SET_FRAC, BAG count effects
# techniques -- BUILD ACTUAL TRENDLINE (so elapsed isn't doing all the work)
# techniques -- LOAD IN SOME SIMPLE RIDGES

# features   -- include population and much more; is_county, is_state, per capita * population etc. etc.


# ### Plot

# In[ ]:





# In[ ]:





# In[ ]:





# ### Extra Code
# %% [code]
first_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').first()
last_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').last()

first_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').first()
last_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').last()

first_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').first()
last_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').last()

# %% [raw]
# test_wp

# %% [raw]
# first_c_slope

# %% [raw]
# test_wp

# %% [raw]
# test_wp

# %% [code]
test_wp.case_slope = np.where(  test_wp.case_slope.isnull() & 
                     (test_wp.Date < first_c_slope.loc[test_wp.Place].Date.values),
                   
                  first_c_slope.loc[test_wp.Place].case_slope.values,
                     test_wp.case_slope
                  )

test_wp.case_slope = np.where(  test_wp.case_slope.isnull() & 
                     (test_wp.Date > last_c_slope.loc[test_wp.Place].Date.values),
                   
                  last_c_slope.loc[test_wp.Place].case_slope.values,
                     test_wp.case_slope
                  )

# %% [code]
test_wp.fatality_slope = np.where(  test_wp.fatality_slope.isnull() & 
                     (test_wp.Date < first_f_slope.loc[test_wp.Place].Date.values),
                   
                  first_f_slope.loc[test_wp.Place].fatality_slope.values,
                     test_wp.fatality_slope
                  )

test_wp.fatality_slope = np.where(  test_wp.fatality_slope.isnull() & 
                     (test_wp.Date > last_f_slope.loc[test_wp.Place].Date.values),
                   
                  last_f_slope.loc[test_wp.Place].fatality_slope.values,
                     test_wp.fatality_slope
                  )

# %% [code]
test_wp.cfr_pred = np.where(  test_wp.cfr_pred.isnull() & 
                     (test_wp.Date < first_cfr_pred.loc[test_wp.Place].Date.values),
                   
                  first_cfr_pred.loc[test_wp.Place].cfr_pred.values,
                     test_wp.cfr_pred
                  )

test_wp.cfr_pred = np.where(  test_wp.cfr_pred.isnull() & 
                     (test_wp.Date > last_cfr_pred.loc[test_wp.Place].Date.values),
                   
                  last_cfr_pred.loc[test_wp.Place].cfr_pred.values,
                     test_wp.cfr_pred
                  )

# %% [code]


# %% [code]
test_wp.case_slope = test_wp.case_slope.interpolate('linear')
test_wp.fatality_slope = test_wp.fatality_slope.interpolate('linear')
test_wp.cfr_pred = test_wp.cfr_pred.interpolate('linear')

# %% [code]
test_wp.case_slope = test_wp.case_slope.fillna(0)
test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

# test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

# %% [raw]
# test_wp.cfr_pred.isnull().sum()

# %% [markdown]
# #### Convert Slopes to Aggregate Counts

# %% [code]
LAST_DATE = test.Date.min() - datetime.timedelta(1)

# %% [code]
final = train_bk[train_bk.Date == LAST_DATE  ]

# %% [raw]
# train

# %% [raw]
# final

# %% [code]
test_wp = pd.merge(test_wp, final[['Place', 'ConfirmedCases', 'Fatalities']], on='Place', 
                   how ='left', validate='m:1')

# %% [raw]
# test_wp

# %% [code]
LAST_DATE

# %% [raw]
# test_wp

# %% [code]
test_wp.ConfirmedCases = np.exp( 
                            np.log(test_wp.ConfirmedCases + 1) \
                                + test_wp.case_slope * 
                                   (test_wp.Date - LAST_DATE).dt.days )- 1

test_wp.Fatalities = np.exp(
                            np.log(test_wp.Fatalities + 1) \
                              + test_wp.fatality_slope * 
                                   (test_wp.Date - LAST_DATE).dt.days )  -1

# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1
                                     

# %% [code]
LAST_DATE

# %% [raw]
# final[final.Place=='Italy']

# %% [code]
test_wp[ (test_wp.Country == 'Italy')].groupby('Date').sum()[:10]


# %% [code]
test_wp[ (test_wp.Country == 'US')].groupby('Date').sum().iloc[-5:]


# %% [code]


# %% [markdown]
# ### Final Merge

# %% [code]
final = train_bk[train_bk.Date == test.Date.min() - datetime.timedelta(1) ]

# %% [code]
final.head()

# %% [code]
test['elapsed'] = (test.Date - final.Date.max()).dt.days 

# %% [raw]
# test.Date

# %% [code]
test.elapsed

# %% [code]


# %% [markdown]# ### CFR Caps

# %% [code]
full_bk = test_wp.copy()

# %% [code]
full = test_wp.copy()

# %% [code]


# %% [code]
BASE_RATE = 0.01

# %% [code]
CFR_CAP = 0.13

# %% [code]


# %% [code]
lplot(full_bk)

# %% [code]
lplot(full_bk, columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [code]
full['cfr_imputed_fatalities_low'] = full.ConfirmedCases * np.exp(full.cfr_pred) / np.exp(0.5)
full['cfr_imputed_fatalities_high'] = full.ConfirmedCases * np.exp(full.cfr_pred) * np.exp(0.5)
full['cfr_imputed_fatalities'] = full.ConfirmedCases * np.exp(full.cfr_pred)  

# %% [code]


# %% [raw]
# full[(full.case_slope > 0.02) & 
#           (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
#                 (full.cfr_imputed_fatalities_low > 0.3) &
#                 ( full.Fatalities < 100 ) &
#     (full.Country!='China')] \
#      .groupby('Place').count()\
#     .sort_values('ConfirmedCases', ascending=False).iloc[:, 9:]

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
                (full.cfr_imputed_fatalities_low > 0.3) &
                ( full.Fatalities < 100000 ) &
    (full.Country!='China') &
     (full.Date == datetime.datetime(2020, 4,15))] \
     .groupby('Place').last()\
    .sort_values('Fatalities', ascending=False).iloc[:, 9:]

# %% [code]
(np.log(full.Fatalities + 1) -np.log(full.cfr_imputed_fatalities) ).plot(kind='hist', bins = 250)

# %% [raw]
# full[  
#                    (np.log(full.Fatalities + 1) < np.log(full.cfr_imputed_fatalities_high + 1) -0.5    ) 
#     & (~full.Country.isin(['China', 'Korea, South']))
#                 ][full.Date==train.Date.max()]\
#      .groupby('Place').first()\
#     .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
                (full.cfr_imputed_fatalities_low > 0.3) &
                ( full.Fatalities < 100000 ) &
    (~full.Country.isin(['China', 'Korea, South']))][full.Date==train.Date.max()]\
     .groupby('Place').first()\
    .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]

# %% [code]
full.Fatalities = np.where(   
    (full.case_slope > 0.00) & 
                   (full.Fatalities <= full.cfr_imputed_fatalities_low    ) &
                (full.cfr_imputed_fatalities_low > 0.0) &
                ( full.Fatalities < 100000 ) &
    (~full.Country.isin(['China', 'Korea, South'])) ,
                        
                        (full.cfr_imputed_fatalities_high + full.cfr_imputed_fatalities)/2,
                                    full.Fatalities)
    

# %% [raw]
# assert len(full) == len(data_wp)

# %% [raw]
# x_test.shape

# %% [code]
full['elapsed'] = (test_wp.Date - LAST_DATE).dt.days

# %% [code]
full[ (full.case_slope > 0.02) & 
          (np.log(full.Fatalities + 1) < np.log(full.ConfirmedCases * BASE_RATE + 1) - 0.5) &
                           (full.Country != 'China')]\
            [full.Date == datetime.datetime(2020, 4, 5)] \
            .groupby('Place').last().sort_values('ConfirmedCases', ascending=False).iloc[:,8:]

# %% [raw]
# full.Fatalities.max()

# %% [code]
full.Fatalities = np.where((full.case_slope > 0.0) & 
                      (full.Fatalities < full.ConfirmedCases * BASE_RATE) &
                           (full.Country != 'China'), 
                                            
            np.exp(   
                    np.log( full.ConfirmedCases * BASE_RATE + 1) \
                           * np.clip(   0.5* (full.elapsed - 1) / 30, 0, 1) \
                           
                     +  np.log(full.Fatalities +1 ) \
                           * np.clip(1 - 0.5* (full.elapsed - 1) / 30, 0, 1)
            ) -1
                           
                           ,
                                               full.Fatalities)  

# %% [raw]
# full.elapsed

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
    (full.Country!='China')]\
     .groupby('Place').count()\
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# %% [raw]
# full[full.Place=='United KingdomTurks and Caicos Islands']

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high * 2   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
    (full.Country!='China')  ]\
     .groupby('Place').last()\
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high * 1.5   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
    (full.Country!='China')][full.Date==train.Date.max()]\
     .groupby('Place').first()\
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# %% [code]


# %% [code]
full.Fatalities =  np.where(  (full.case_slope > 0.0) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high      * 2   ) &
                (full.cfr_imputed_fatalities_low > 0.0) &
                (full.Country!='China') ,
                            
                     full.cfr_imputed_fatalities,
                            
                            full.Fatalities)

full.Fatalities =  np.where(  (full.case_slope > 0.0) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high   ) &
                (full.cfr_imputed_fatalities_low > 0.0) &
                (full.Country!='China') ,
                    np.exp(        
                            0.6667 * np.log(full.Fatalities + 1) \
                        + 0.3333 * np.log(full.cfr_imputed_fatalities + 1)
                                ) - 1,
                            
                            full.Fatalities)

# %% [code]


# %% [code]
full[(full.Fatalities > full.ConfirmedCases * CFR_CAP) &
                                          (full.ConfirmedCases > 1000)

    ]                        .groupby('Place').last().sort_values('Fatalities', ascending=False)

# %% [raw]
# full.Fatalities =  np.where( (full.Fatalities > full.ConfirmedCases * CFR_CAP) &
#                                           (full.ConfirmedCases > 1000)
#                                         , 
#                              full.ConfirmedCases * CFR_CAP\
#                                            * np.clip((full.elapsed - 5) / 15, 0, 1) \
#                                  +  full.Fatalities * np.clip(1 - (full.elapsed - 5) / 15, 0, 1)
#                             , 
#                                                full.Fatalities)

# %% [raw]
# train[train.Country=='Italy']

# %% [raw]
# final[final.Country=='US'].sum()

# %% [code]
(np.log(full.Fatalities + 1) -np.log(full.cfr_imputed_fatalities) ).plot(kind='hist', bins = 250)

# %% [code]


# %% [markdown]# ### Fix Slopes now

# %% [raw]
# final

# %% [code]
assert len(pd.merge(full, final, on='Place', suffixes = ('', '_i'), validate='m:1')) == len(full)

# %% [code]
ffm = pd.merge(full, final, on='Place', suffixes = ('', '_i'), validate='m:1')
ffm['fatality_slope'] = (np.log(ffm.Fatalities + 1 )\
                             - np.log(ffm.Fatalities_i + 1 ) ) \
                                 / ffm.elapsed
ffm['case_slope'] = (np.log(ffm.ConfirmedCases + 1 ) \
                             - np.log(ffm.ConfirmedCases_i + 1 ) ) \
                                 / ffm.elapsed

# %% [markdown]
# #### Fix Upward Slopers

# %% [raw]
# final_slope = (ffm.groupby('Place').last().case_slope)
# final_slope.sort_values(ascending=False)
# 
# high_final_slope = final_slope[final_slope > 0.1].index

# %% [raw]
# slope_change = (ffm.groupby('Place').last().case_slope - ffm.groupby('Place').first().case_slope)
# slope_change.sort_values(ascending = False)
# high_slope_increase = slope_change[slope_change > 0.05].index

# %% [code]


# %% [raw]
# test.Date.min()

# %% [raw]
# set(high_slope_increase) & set(high_final_slope)

# %% [raw]
# ffm.groupby('Date').case_slope.median()

# %% [code]


# %% [markdown]# ### Fix Drop-Offs

# %% [code]
ffm[np.log(ffm.Fatalities+1) < np.log(ffm.Fatalities_i+1) - 0.2]\
    [['Place', 'Date', 'elapsed', 'Fatalities', 'Fatalities_i']]

# %% [code]
ffm[np.log(ffm.ConfirmedCases + 1) < np.log(ffm.ConfirmedCases_i+1) - 0.2]\
    [['Place', 'elapsed', 'ConfirmedCases', 'ConfirmedCases_i']]

# %% [code]


# %% [raw]
# (ffm.groupby('Place').last().fatality_slope - ffm.groupby('Place').first().fatality_slope)\
#     .sort_values(ascending = False)[:10]

# %% [markdown]
# ### Display

# %% [raw]
# full[full.Country=='US'].groupby('Date').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'mean',
#         'fatality_slope': 'mean',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     })

# %% [code]
full_bk[(full_bk.Date == test.Date.max() ) & 
   (~full_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)

# %% [raw]
# full[full.Country=='China'].groupby('Date').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'mean',
#         'fatality_slope': 'mean',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     })[::5]

# %% [code]


# %% [raw]
# ffc = pd.merge(final, full, on='Place', validate = '1:m')
# ffc[(np.log(ffc.Fatalities_x) - np.log(ffc.ConfirmedCase_x)) / ffc.elapsed_y ]

# %% [raw]
# ffm.groupby('Place').case_slope.last().sort_values(ascending = False)[:30]

# %% [raw]
# lplot(test_wp)

# %% [raw]
# lplot(test_wp, columns = ['case_slope', 'fatality_slope'])

# %% [code]

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)])

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [raw]
# test.Date.min()

# %% [code]
ffm.fatality_slope = np.clip(ffm.fatality_slope, None, 0.5)

# %% [raw]
# ffm.case_slope = np.clip(ffm.case_slope, None, 0.25)

# %% [code]


# %% [raw]
# for lr in [0.05, 0.02, 0.01, 0.007, 0.005, 0.003]:
# 
#     ffm.loc[ (ffm.Place==ffm.Place.shift(1) )
#          & (ffm.Place==ffm.Place.shift(-1) ) &
#      ( np.abs ( (ffm.case_slope.shift(-1) + ffm.case_slope.shift(1) ) / 2
#                        - ffm.case_slope).fillna(0)
#                     > lr ), 'case_slope'] = \
#                      ( ffm.case_slope.shift(-1) + ffm.case_slope.shift(1) ) / 2
# 

# %% [code]
for lr in [0.2, 0.14, 0.1, 0.07, 0.05, 0.03, 0.01 ]:

    ffm.loc[ (ffm.Place==ffm.Place.shift(4) )
         & (ffm.Place==ffm.Place.shift(-4) ), 'fatality_slope'] = \
         ( ffm.fatality_slope.shift(-2) * 0.25 \
              + ffm.fatality_slope.shift(-1) * 0.5 \
                + ffm.fatality_slope \
                  + ffm.fatality_slope.shift(1) * 0.5 \
                    + ffm.fatality_slope.shift(2) * 0.25 ) / 2.5


# %% [code]


# %% [code]
ffm.ConfirmedCases = np.exp( 
                            np.log(ffm.ConfirmedCases_i + 1) \
                                + ffm.case_slope * 
                                   ffm.elapsed ) - 1

ffm.Fatalities = np.exp(
                            np.log(ffm.Fatalities_i + 1) \
                              + ffm.fatality_slope * 
                                   ffm.elapsed ) - 1
# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1
                                     

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)])

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [code]
ffm[(ffm.Date == test.Date.max() ) & 
   (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)

# %% [code]


# %% [code]
ffm_bk = ffm.copy()

# %% [code]


# %% [code]


# %% [code]
ffm = ffm_bk.copy()

# %% [code]
counter = Counter(data.Place)
# counter.most_common()
median_count = np.median([ counter[group] for group in ffm.Place])
# [ (group, np.round( np.power(counter[group] / median_count, -1),3) ) for group in 
#      counter.keys()]
c_count = [ np.clip(
            np.power(counter[group] / median_count, -1.5), None, 2.5) for group in ffm.Place]
 

# %% [code]
RATE_MULT = 0.00
RATE_ADD = 0.003
LAG_FALLOFF = 15

ma_factor = np.clip( ( ffm.elapsed - 14) / 14 , 0, 1)

ffm.case_slope = np.where(ffm.elapsed > 0,
    0.7 * ffm.case_slope * (1+ ma_factor * RATE_MULT) \
         + 0.3 * (  ffm.case_slope.ewm(span=LAG_FALLOFF).mean()\
                                                      * np.clip(ma_factor, 0, 1)
                      + ffm.case_slope    * np.clip( 1 - ma_factor, 0, 1)) 
                          
                          + RATE_ADD * ma_factor * c_count,
         ffm.case_slope)

# --

RATE_MULT = 0
RATE_ADD = 0.015
LAG_FALLOFF = 15

ma_factor = np.clip( ( ffm.elapsed - 10) / 14 , 0, 1)


ffm.fatality_slope = np.where(ffm.elapsed > 0,
    0.3 * ffm.fatality_slope * (1+ ma_factor * RATE_MULT) \
         + 0.7* (  ffm.fatality_slope.ewm(span=LAG_FALLOFF).mean()\
                                                              * np.clip( ma_factor, 0, 1)
                      + ffm.fatality_slope    * np.clip( 1 - ma_factor, 0, 1)   )
                              
                              + RATE_ADD * ma_factor * c_count \
                              
                              
                              * (ffm.Country != 'China')
                              ,
         ffm.case_slope)

# %% [code]
ffm.ConfirmedCases = np.exp( 
                            np.log(ffm.ConfirmedCases_i + 1) \
                                + ffm.case_slope * 
                                   ffm.elapsed ) - 1

ffm.Fatalities = np.exp(
                            np.log(ffm.Fatalities_i + 1) \
                              + ffm.fatality_slope * 
                                   ffm.elapsed ) - 1
# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1
                                     

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)])

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [raw]
# LAST_DATE

# %% [code]
ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
   (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[:15]

# %% [code]
ffm[(ffm.Date == test.Date.max() ) & 
   (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[:15]

# %% [code]


# %% [code]


# %% [code]


# %% [code]
ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
   (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[-50:]

# %% [code]
ffm[(ffm.Date == test.Date.max() ) & 
   (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).loc[ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
   (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[-50:].index]

# %% [code]


# %% [code]
# use country-specific CFR !!!!  helps cap US and raise up Italy !
# could also use lagged CFR off cases as of 2 weeks ago...
 # ****  keep everything within ~0.5 order of magnitude of its predicted CFR.. !!


# %% [code]


# %% [markdown]
# ### Join

# %% [raw]
# assert len(test_wp) == len(full)
# 

# %% [raw]
# full = pd.merge(test_wp, full[['Place', 'Date', 'Fatalities']], on = ['Place', 'Date'],
#             validate='1:1')

# %% [code]


# %% [markdown]# ### Fill in New Places with Ramp Average

# %% [code]
NUM_TEST_DATES = len(test.Date.unique())

base = np.zeros((2, NUM_TEST_DATES))
base2 = np.zeros((2, NUM_TEST_DATES))

# %% [code]
for idx, c in enumerate(['ConfirmedCases', 'Fatalities']):
    for n in range(0, NUM_TEST_DATES):
        base[idx,n] = np.mean(
            np.log(  train[((train.Date < test.Date.min())) & 
              (train.ConfirmedCases > 0)].groupby('Country').nth(n)[c]+1))

# %% [code]
base = np.pad( base, ((0,0), (6,0)), mode='constant', constant_values = 0)

# %% [code]
for n in range(0, base2.shape[1]):
    base2[:, n] = np.mean(base[:, n+0: n+7], axis = 1)

# %% [code]
new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &
      (train.ConfirmedCases == 0)
     ].Place

# %% [code]
# fill in new places 
ffm.ConfirmedCases = \
    np.where(   ffm.Place.isin(new_places),
          base2[ 0, (ffm.Date - test.Date.min()).dt.days],
                 ffm.ConfirmedCases)
ffm.Fatalities = \
    np.where(   ffm.Place.isin(new_places),
          base2[ 1, (ffm.Date - test.Date.min()).dt.days],
                 ffm.Fatalities)

# %% [code]


# %% [code]
ffm[ffm.Country=='US'].groupby('Date').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
        'fatality_slope': 'mean',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    })

# %% [raw]
# train[train.Country == 'US'].Province_State.unique()

# %% [markdown]
# ### Save

# %% [code]


# %% [code]


# %% [code]
sub = pd.read_csv(input_path + '/submission.csv')

# %% [code]
scl = sub.columns.to_list()

# %% [code]

# print(full_bk.groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])
# print(ffm.groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])


# %% [code]
if ffm[scl].isnull().sum().sum() == 0:
    out = full_bk[scl] * 0.0 + ffm_bk[scl] * 0.5 + full[scl] * 0.5 + ffm[scl] * 0.0
else:
    print('using full-bk')
    out = full_bk[scl]

out.ForecastId = np.round(out.ForecastId, 0).astype(int) 

print(pd.merge(out, test[['ForecastId', 'Date', 'Place']], on='ForecastId')\
      .sort_values('ForecastId')\
          .groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])

out = np.round(out, 2)
private = out[sub.columns.to_list()]
  # load public LB results
tt = pd.merge(train, test, on=['Place', 'Date'], 
              how='right', validate="1:1")\
                    .fillna(method = 'ffill')
public = tt[['ForecastId', 'ConfirmedCases', 'Fatalities']]# concat public and private predictions;
full_pred = pd.concat((private, public[~public.ForecastId.isin(private.ForecastId)]),
     ignore_index=True).sort_values('ForecastId')

full_pred.to_csv('submission.csv', index=False)
# In[ ]:





# In[ ]:




# %% [markdown]
# ### Train Fix

# %% [markdown]
# #### Supplement Missing US Data

# %% [code]
revised = pd.read_csv(path + '/outside_data' + 
                          '/covid19_train_data_us_states_before_march_09_new.csv')


# %% [raw]
# revised.Date = pd.to_datetime(revised.Date)
# revised.Date = revised.Date.apply(datetime.datetime.strftime, args= ('%Y-%m-%d',))

# %% [code]
revised = revised[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']]

# %% [code]
train.tail()

# %% [code]
revised.head()

# %% [code]
train.Date = pd.to_datetime(train.Date)
revised.Date = pd.to_datetime(revised.Date)

# %% [code]
rev_train = pd.merge(train, revised, on=['Province_State', 'Country_Region', 'Date'],
                            suffixes = ('', '_r'), how='left')

# %% [code]


# %% [code]
rev_train[~rev_train.ConfirmedCases_r.isnull()].head()

# %% [code]


# %% [code]


# %% [code]
rev_train.ConfirmedCases = \
    np.where( (rev_train.ConfirmedCases == 0) & ((rev_train.ConfirmedCases_r > 0 )) &
                 (rev_train.Country_Region == 'US'),
        
        rev_train.ConfirmedCases_r,
            rev_train.ConfirmedCases)


# %% [code]
rev_train.Fatalities = \
    np.where( ~rev_train.Fatalities_r.isnull() & 
                (rev_train.Fatalities == 0) & ((rev_train.Fatalities_r > 0 )) &
                 (rev_train.Country_Region == 'US')
             ,
        
        rev_train.Fatalities_r,
            rev_train.Fatalities)


# %% [code]
rev_train.drop(columns = ['ConfirmedCases_r', 'Fatalities_r'], inplace=True)

# %% [code]
train = rev_train

# %% [raw]
# train[train.Province_State == 'California']

# %% [raw]
# import sys
# def sizeof_fmt(num, suffix='B'):
#     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f %s%s" % (num, 'Yi', suffix)
# 
# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key= lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
# # ### Oxford Actions Database

# %% [code]
# contain_data = pd.read_excel(path + '/outside_data' + 
#                           '/OxCGRT_Download_latest_data.xlsx')

contain_data = pd.read_csv(path + '/outside_data' + 
                          '/OxCGRT_Download_070420_160027_Full.csv')

# %% [code] {"scrolled":true}
contain_data = contain_data[[c for c in contain_data.columns if 
                      not any(z in c for z in ['_Notes','Unnamed', 'Confirmed',
                                               'CountryCode',
                                                      'S8', 'S9', 'S10','S11',
                                              'StringencyIndexForDisplay'])] ]\
        

# %% [code]
contain_data.rename(columns = {'CountryName': "Country"}, inplace=True)

# %% [code]
contain_data.Date = contain_data.Date.astype(str)\
    .apply(datetime.datetime.strptime, args=('%Y%m%d', ))

# %% [code]


# %% [code]
contain_data_orig = contain_data.copy()

# %% [code]
contain_data.columns

# %% [raw]
# contain_data.columns

# %% [code]


# %% [code]
cds = []
for country in contain_data.Country.unique():
    cd = contain_data[contain_data.Country==country]
    cd = cd.fillna(method = 'ffill').fillna(0)
    cd.StringencyIndex = cd.StringencyIndex.cummax()  # for now
    col_count = cd.shape[1]
    
    # now do a diff columns
    # and ewms of it
    for col in [c for c in contain_data.columns if 'S' in c]:
        col_diff = cd[col].diff()
        cd[col+"_chg_5d_ewm"] = col_diff.ewm(span = 5).mean()
        cd[col+"_chg_20_ewm"] = col_diff.ewm(span = 20).mean()
        
    # stringency
    cd['StringencyIndex_5d_ewm'] = cd.StringencyIndex.ewm(span = 5).mean()
    cd['StringencyIndex_20d_ewm'] = cd.StringencyIndex.ewm(span = 20).mean()
    
    cd['S_data_days'] =  (cd.Date - cd.Date.min()).dt.days
    for s in [1, 10, 20, 30, 50, ]:
        cd['days_since_Stringency_{}'.format(s)] = \
                np.clip((cd.Date - cd[(cd.StringencyIndex > s)].Date.min()).dt.days, 0, None)
    
    
    cds.append(cd.fillna(0)[['Country', 'Date'] + cd.columns.to_list()[col_count:]])
contain_data = pd.concat(cds)

# %% [raw]
# contain_data.columns

# %% [raw]
# dataset.groupby('Country').S_data_days.max().sort_values(ascending = False)[-30:]

# %% [raw]
# contain_data.StringencyIndex.cummax()

# %% [raw]
# contain_data.groupby('Date').count()[90:]

# %% [code]
contain_data.Date.max()

# %% [code]
contain_data.columns

# %% [code]
contain_data[contain_data.Country == 'Australia']

# %% [code]
contain_data.shape

# %% [raw]
# contain_data.groupby('Country').Date.max()[:50]

# %% [code]
contain_data.Country.replace({ 'United States': "US",
                                 'South Korea': "Korea, South",
                                    'Taiwan': "Taiwan*",
                              'Myanmar': "Burma", 'Slovak Republic': "Slovakia",
                                  'Czech Republic': 'Czechia',

}, inplace=True)

# %% [code]
set(contain_data.Country) - set(test.Country_Region)

# %% [code]# #### Load in Supplementary Data

# %% [code]
sup_data = pd.read_excel(path + '/outside_data' + 
                          '/Data Join - Copy1.xlsx')


# %% [code]
sup_data.columns = [c.replace(' ', '_') for c in sup_data.columns.to_list()]

# %% [code]
sup_data.drop(columns = [c for c in sup_data.columns.to_list() if 'Unnamed:' in c], inplace=True)

# %% [code]


# %% [code]

# %% [raw]
# sup_data.drop(columns = ['longitude', 'temperature', 'humidity',
#                         'latitude'], inplace=True)

# %% [raw]
# sup_data.columns

# %% [raw]
# sup_data.drop(columns = [c for c in sup_data.columns if 
#                                  any(z in c for z in ['state', 'STATE'])], inplace=True)

# %% [raw]
# sup_data = sup_data[['Province_State', 'Country_Region',
#                      'Largest_City',
#                      'IQ', 'GDP_region', 
#                      'TRUE_POPULATION', 'pct_in_largest_city', 
#                    'Migrant_pct',
#                     'Avg_age',
#                      'latitude', 'longitude',
#                 'abs_latitude', #  'Personality_uai', 'Personality_ltowvs',
#               'Personality_pdi',
# 
#                  'murder',  'real_gdp_growth'
#                     ]]

# %% [raw]
# sup_data = sup_data[['Province_State', 'Country_Region',
#                      'Largest_City',
#                      'IQ', 'GDP_region', 
#                      'TRUE_POPULATION', 'pct_in_largest_city', 
#                    #'Migrant_pct',
#                     # 'Avg_age',
#                      # 'latitude', 'longitude',
#              #    'abs_latitude', #  'Personality_uai', 'Personality_ltowvs',
#             #   'Personality_pdi',
# 
#                  'murder', # 'real_gdp_growth'
#                     ]]

# %% [code]
sup_data.drop(columns = [ 'Date', 'ConfirmedCases',
       'Fatalities', 'log-cases', 'log-fatalities', 'continent'], inplace=True)

# %% [raw]
# sup_data.drop(columns = [ 'Largest_City',  
#                         'continent_gdp_pc', 'continent_happiness', 'continent_generosity',
#        'continent_corruption', 'continent_Life_expectancy', 'TRUE_CHINA',
#                          'Happiness', 'Logged_GDP_per_capita',
#        'Social_support','HDI', 'GDP_pc', 'pc_GDP_PPP', 'Gini',
#                          'state_white', 'state_white_asian', 'state_black',
#        'INNOVATIVE_STATE','pct_urban', 'Country_pop', 
#                         
#                         ], inplace=True)

# %% [raw]
# sup_data.columns

# %% [raw]
# 

# %% [code]
sup_data['Migrants_in'] = np.clip(sup_data.Migrants, 0, None)
sup_data['Migrants_out'] = -np.clip(sup_data.Migrants, None, 0)
sup_data.drop(columns = 'Migrants', inplace=True)

# %% [raw]
# sup_data.loc[:, 'Largest_City'] = np.log(sup_data.Largest_City + 1)

# %% [code]
sup_data.head()

# %% [code]


# %% [code]
sup_data.shape

# %% [raw]
# sup_data.loc[4][:50]

# %% [code]

# %% [code]
sup_data['Place'] = sup_data.Country +  sup_data.Province_State.fillna("")

# %% [code]
len(train.Place.unique())

# %% [code]
sup_data = sup_data[    
    sup_data.columns.to_list()[2:]]

# %% [code]
sup_data = sup_data.replace('N.A.', np.nan).fillna(-0.5)

# %% [code]
for c in sup_data.columns[:-1]:
    m = sup_data[c].max() #- sup_data 
    
    if m > 300 and c!='TRUE_POPULATION':
        print(c)
        sup_data[c] = np.log(sup_data[c] + 1)
        assert sup_data[c].min() > -1

# %% [code]
for c in sup_data.columns[:-1]:
    m = sup_data[c].max() #- sup_data 
    
    if m > 300:
        print(c)

# %% [code]


# %% [code]
# In[ ]:




