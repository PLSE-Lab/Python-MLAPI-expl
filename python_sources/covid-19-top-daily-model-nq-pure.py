#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Week 5 

# ### Initialize

# In[ ]:


import pandas as pd
import numpy as np
import os, sys, gc
import psutil

from collections import Counter
from random import shuffle
import math
from scipy.stats.mstats import gmean
import datetime

import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns

pd.options.display.float_format = '{:.8}'.format
plt.rcParams["figure.figsize"] = (17, 5.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
 
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

pd.options.display.max_rows = 100


# In[ ]:


LAST_TRAIN_DATE = datetime.datetime(2020, 5, 10)


# ### Load

# In[ ]:


stack_path = '/kaggle/input/locknquays'
stacks = []
files = os.listdir(stack_path)
list.sort(files)
# print(files)

files = ["wm_true.csv"]
# print(files)


# In[ ]:


PRIVATE_START = '2020-05-13'

DATA_DIR = '/kaggle/input/locknquays/'

top_submission_files = [f for f in os.listdir(DATA_DIR) if 'wm_true2.csv' not in f and 'weighted_mean' not in f]
files = top_submission_files
list.sort(files)
# files


# In[ ]:


stacks = []
for file in files:
    df = pd.read_csv(DATA_DIR +file, index_col = 'ForecastId_Quantile')
    df.rename(columns = {'TargetValue':"preds_{}".format(file)}, inplace=True)
    stacks.append(df)


# In[ ]:


stacks.append(pd.read_csv('/kaggle/input/covid-19-daily-counts-a3-bag-1/submission.csv', index_col = 'ForecastId_Quantile')        .rename(columns = {'TargetValue':"preds_{}".format('NQ_A3_bag1.csv')}))


# In[ ]:


stack = stacks[0]
for new_stack in stacks[1:]:
    stack = stack.merge(new_stack, left_index=True, right_index=True)


# In[ ]:


stack.rename(columns = {'preds_c19-wk5-nq-median.csv': 'preds_nq_model.csv',
                           'preds_wm_true.csv': 'preds_weighted_mean_1.csv',
                           'preds_wm_true2fix.csv': 'preds_weighted_mean_2.csv',
                       }, inplace=True)


# In[ ]:


stack['preds_rw_model.csv'] = stack['preds_simple_mean.csv'] * 2 - stack['preds_nq_model.csv']
stack.drop(columns = 'preds_simple_mean.csv', inplace=True)


# In[ ]:


N_STACKS = len([c for c in stack if 'preds_' in c])


# ### Join with Test

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

train['Place'] = train.Country + ('_' + train.Province_State).fillna("") +  ('_' + train.County ).fillna("")
test['Place'] = test.Country +  ('_' + test.Province_State).fillna("") + ('_' + test.County).fillna("") 

train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)

train_bk = train.copy()
full_train = train.copy()


# In[ ]:


pdt = ['Place', 'Date', 'Target']

stack['ForecastId'] = stack.index.str.split('_').str[0].astype(int)
stack['Quantile'] = stack.index.str.split('_').str[1]


# In[ ]:


MIN_DATE = test.Date.min() # datetime.datetime(2020, 3, 1)


# In[ ]:


full_stack = stack.merge(test, on='ForecastId', validate='m:1')    .merge(train, on=pdt, suffixes=('', '_y'), how='outer')
full_stack = full_stack[full_stack.Date >= MIN_DATE]


# In[ ]:


def loss(preds, actual, weights, qs):
    l = 1 * (actual >= preds) * qs * (actual - preds) - 1 * (actual < preds) * (1 - qs) * (actual - preds)
    return l * weights


# In[ ]:


top_submissions = full_stack[~full_stack.TargetValue.isnull() 
                                & (full_stack.Date >= PRIVATE_START)
                                    & (full_stack.Date <= train.Date.max())].copy()
top_submissions['q'] = top_submissions.Quantile.astype(np.float)
predictions = top_submissions.iloc[:,:N_STACKS]
for p in predictions:
    top_submissions[f'loss_{p}'] = loss(top_submissions[f'{p}'],
                                        top_submissions.TargetValue,
                                        top_submissions.Weight,
                                        top_submissions.q)


# In[ ]:


def strip(x):
    return x.split('loss_preds_')[-1]


# In[ ]:


np.round(top_submissions.rename(columns = strip).groupby('Date').mean().iloc[:, -N_STACKS:], 4).plot(title = 'Daily Loss');


# In[ ]:


scores = top_submissions.iloc[:, -N_STACKS:]
scores.columns = scores.columns.str.split('preds_').str[1].str.slice(None,-4)
scores = np.round(np.mean(scores).sort_values(), 4)


# In[ ]:


score_df = pd.DataFrame(scores, columns = ['score'])
score_df['rank'] = score_df.score.rank()
print('Rankings')
score_df.score


# In[ ]:


non_zero = full_stack[ full_stack.Date >= PRIVATE_START]

nz_clean = non_zero.copy()
nz_clean.columns = nz_clean.columns.str.replace('preds_', '').str.slice(None, -4)
print('Correlation Matrix')
np.round(nz_clean[scores.index].corr(),3)


# In[ ]:


top_preds = ['preds_{}.csv'.format(n) for n in score_df.index]


# In[ ]:


full_stack = full_stack[top_preds + [c for c in full_stack if c not in top_preds]]


# In[ ]:


full_stack['elapsed'] = (full_stack.Date - full_stack.Date.min()).dt.days +1

preds = full_stack.iloc[:, :N_STACKS]

# full_stack['mean_pred'] = np.mean(preds, axis = 1)
# full_stack['stdev_pred'] = np.std(preds, axis = 1)
# full_stack['mean_over_stdev_pred'] = (full_stack.mean_pred / full_stack.stdev_pred ).fillna(-1)

# full_stack['mean_log_pred'] = np.mean( np.log( 1 + np.clip(preds, 0, None)), axis = 1)
# full_stack['stdev_log_pred'] = np.std( np.log( 1 + np.clip(preds, 0, None)), axis = 1)
# full_stack['mean_over_stdev_log_pred'] = ( full_stack.mean_log_pred / full_stack.stdev_log_pred ).fillna(-1)


# In[ ]:


full_stack[(full_stack.Place == 'US_New York') & (full_stack.Date >= PRIVATE_START)
              & (full_stack.Quantile == '0.5') & (full_stack.Target == 'ConfirmedCases')
          ].set_index(['Date'])


# ### Basics

# In[ ]:


pdt = ['Place', 'Date', 'Target']

population = train.groupby('Place').Population.mean()
state = train.groupby('Place').Province_State.first()

train_avg = train.groupby(['Place', 'Date', 'Target']).mean()



train_pivot_cc = train_avg[train_avg.index.get_level_values(2)=='ConfirmedCases'].reset_index()    .pivot('Date', 'Place', 'TargetValue')

train_pivot_f = train_avg[train_avg.index.get_level_values(2)=='Fatalities'].reset_index()    .pivot('Date', 'Place', 'TargetValue')


# ### Begin Basic Features

# In[ ]:


data_cc = train_pivot_cc
data_fs = train_pivot_f

def columize(pivot_df):
    return pivot_df.reset_index().melt(id_vars = 'Date', value_name = 'Value').Value


# In[ ]:


full_stack['place_type'] = full_stack.Place.str.split('_').apply(len)


# ### Calc Features

# In[ ]:


dataset = data_cc.astype(np.float32).reset_index().melt(id_vars='Date', value_name = 'ConfirmedCases')
dataset = dataset.merge(data_fs.astype(np.float32).reset_index()                            .melt(id_vars='Date', value_name = 'Fatalities'),
                        on = ['Date', 'Place'])

dataset = dataset[dataset.Date == dataset.Date.max()].iloc[:,:2]

CC = 'ConfirmedCases'
FS = 'Fatalities'

train_cc = train[train.Target==CC].sort_values(['Place', 'Date'])
train_fs = train[train.Target==FS].sort_values(['Place', 'Date'])

total_cc = data_cc.cumsum()
total_fs = data_fs.cumsum()


# In[ ]:


date_total_cc = np.sum(total_cc, axis = 1)
date_total_fs = np.sum(total_fs, axis = 1)


# ### Data Cleanup

# In[ ]:


full_stack_clean = full_stack.drop(columns = ['ForecastId', 'Id', 'Weight'] + 
                        [c for c in full_stack if '_y' in c]  ).fillna('None')


# In[ ]:


UNIQUE_COLS = full_stack_clean.columns[full_stack_clean.groupby(['Place', 'Date']).nunique().mean() > 1]


joined_data = full_stack_clean.groupby(['Place', 'Date']).first().reset_index()    [[c for c in full_stack_clean.columns if c not in UNIQUE_COLS]]


# In[ ]:


UNIQUE_COLS = UNIQUE_COLS.drop(['Target', 'Quantile'])


# In[ ]:


for target in ['ConfirmedCases', 'Fatalities']:
    joined_data = joined_data.merge( full_stack[full_stack.Target == target]                                        .groupby(['Place', 'Date']).TargetValue.first().rename(target),
                                        on = ['Place', 'Date'])
    for quantile in full_stack.Quantile.unique()[:3]:
        df = full_stack[ (full_stack.Target == target) 
                                                     & (full_stack.Quantile == quantile)]\
                                        [list(UNIQUE_COLS) + ['Date', 'Place']].drop(columns = 'TargetValue')
                                                
        df.columns = [ c if ((c=='Date') | (c== 'Place')) else
                              c + '_' + target[0].lower() + '_' + quantile 
                             for c in df.columns]
        joined_data = joined_data.merge(df ,
                                        on = ['Place', 'Date'])
        


# In[ ]:


joined_data = joined_data.sort_values(['Date', 'Place'])

assert (joined_data.isnull().sum() > 0).sum() <=3


# In[ ]:


data_test = test


# In[ ]:


q_labels = [ '5','', '95'];
data_nq = joined_data.copy()
for target in ['ConfirmedCases', 'Fatalities']:
    for idx, quantile in enumerate(full_stack.Quantile.unique()[:3]):
            data_nq[target+ q_labels[idx] + '_pred'] = 0
            

            data_nq[target+ q_labels[idx] + '_pred'] =                    data_nq[[      c + '_' + target[0].lower() + '_' + quantile 
                                     for c in stack[top_preds].columns]] # .iloc[:,:-1] 


# In[ ]:


if len(test) > 0:
    base_date = test.Date.min() - datetime.timedelta(1)
else:
    base_date = train.Date.max()


# ### Get Predictions Merged with Test/Train Data

# In[ ]:


cols = [ c for c in data_nq.columns if 'preds_' in c] 


# In[ ]:


test_nq = pd.merge(test, data_nq[['Date', 'Place'] + 
                                 cols +
                                 ['elapsed']], 
            how='outer', on = ['Date', 'Place'])
test_nq = test_nq[test_nq.Date >= MIN_DATE]


# In[ ]:


cs = ['ConfirmedCases', 'Fatalities']


# In[ ]:


matched_nq = pd.merge(test_nq[pdt + [c for c in test_nq if any(z in c for z in cols)]].dropna(),
             train_bk,
                    on = pdt, how='outer')
matched_nq = matched_nq[matched_nq.Date >= MIN_DATE]


# In[ ]:


places = (data_cc.ewm(span = 20).mean().iloc[-1,:]    + 10 * data_fs.ewm(span = 20).mean().iloc[-1,:] ).sort_values(ascending=False).index[:30]


# ### Strip Early Predictions

# In[ ]:


for col in [c for c in cols if '_c_' in c]:
    old = (matched_nq.Target=='ConfirmedCases') &                     (~matched_nq.TargetValue.isnull()) &                         (matched_nq.Date <= LAST_TRAIN_DATE)
    matched_nq.loc[old,col] =             matched_nq.loc[old].TargetValue
    matched_nq[col].fillna(method='ffill', inplace=True)
for col in [c for c in cols if '_f_' in c]:
    old = (matched_nq.Target=='Fatalities') &                     (~matched_nq.TargetValue.isnull()) &                         (matched_nq.Date <= LAST_TRAIN_DATE)
    matched_nq.loc[old,col] =             matched_nq.loc[old].TargetValue
    matched_nq[col].fillna(method='bfill', inplace=True)


# ### Strip Late 0's

# In[ ]:


matched_nq.loc[matched_nq.Date > train.Date.max(), 'TargetValue'] = np.nan


# ### Final

# In[ ]:


matched_final = matched_nq
matched_final.columns = matched_final.columns.str.replace('preds_', '').str.replace('.csv', '')
cols = [ c.replace('preds_', '').replace('.csv', '') for c in cols] 


# In[ ]:


plt.rcParams.update({'figure.max_open_warning': 0})


# ###  

# ###   

# ## Worldwide Rollup

# In[ ]:


lt = 3.5; lm = 1.6; lq = 1.1; llq = 0.8;


# In[ ]:


rank_pwr = 0.5;


# In[ ]:


l1 = (matched_nq.Place.str.split('_').apply(len) == 1) & (matched_final.Target=='ConfirmedCases')
place = 'Worldwide Roll-up'

ax = matched_final[l1 & (matched_nq.Date <= train.Date.max())]        .groupby('Date').sum()        [["TargetValue"] ]    .plot(title = '{} - Confirmed Cases'.format(place), 
          linewidth = lt, legend=True);

for col in [c for c in cols if '_c_0.5' in c]:
    rank = score_df.loc[col.split('_c_0.')[0]]['rank']
    ax = matched_final[l1 ]            .groupby('Date').sum()            [col]        .plot(title = '{} - Confirmed Cases'.format(place), ax = ax, 
              linewidth = lm * rank**-rank_pwr, legend=True);
    
for col in [c for c in cols if '_c_0.95' in c]:
    rank = score_df.loc[col.split('_c_0.')[0]]['rank']
    ax = matched_final[l1 ]            .groupby('Date').sum()            [col]        .plot(title = '{} - Confirmed Cases'.format(place), ax = ax, 
              linewidth = lq * rank**-rank_pwr, linestyle = '-.', legend=True);

for col in [c for c in cols if '_c_0.05' in c]:
    rank = score_df.loc[col.split('_c_0.')[0]]['rank']
    ax = matched_final[l1 ]            .groupby('Date').sum()            [col]        .plot(title = '{} - Confirmed Cases'.format(place), ax = ax, 
              linewidth = llq * rank**-rank_pwr, linestyle = '-.', legend=True);


    #     break


# In[ ]:


l1 = (matched_nq.Place.str.split('_').apply(len) == 1) & (matched_final.Target=='Fatalities')
place = 'Worldwide Roll-up'

ax = matched_final[l1 & (matched_nq.Date <= train.Date.max())]        .groupby('Date').sum()        [["TargetValue"] ]    .plot(title = '{} - Fatalities'.format(place), 
          linewidth = lt, legend=True);

for col in [c for c in cols if '_f_0.5' in c]:
    rank = score_df.loc[col.split('_f_0.')[0]]['rank']
    ax = matched_final[l1 ]            .groupby('Date').sum()            [col]        .plot( ax = ax, 
              linewidth = lm * rank**-rank_pwr, legend=True);
    
for col in [c for c in cols if '_f_0.95' in c]:
    rank = score_df.loc[col.split('_f_0.')[0]]['rank']
    ax = matched_final[l1 ]            .groupby('Date').sum()            [col]        .plot( ax = ax, 
              linewidth = lq * rank**-rank_pwr, linestyle = '-.', legend=True);

for col in [c for c in cols if '_f_0.05' in c]:
    rank = score_df.loc[col.split('_f_0.')[0]]['rank']
    ax = matched_final[l1 ]            .groupby('Date').sum()            [col]        .plot( ax = ax, 
              linewidth = llq * rank**-rank_pwr, linestyle = '-.', legend=True);


# ###   

# ## Cases

# In[ ]:


for place in places:
    
    
    ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='ConfirmedCases')]            .set_index('Date')            [["TargetValue"] ]        .plot(title = '{} - Confirmed Cases'.format(place), 
              linewidth = lt, legend=True);
    
    for col in [c for c in cols if '_c_0.5' in c]:
        rank = score_df.loc[col.split('_c_0.')[0]]['rank']
        ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='ConfirmedCases')]                .set_index('Date')                [col]            .plot(ax = ax, 
                  linewidth = lm * rank**-rank_pwr, legend=True);
        
    for col in [c for c in cols if '_c_0.95' in c]:
        rank = score_df.loc[col.split('_c_0.')[0]]['rank']
        ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='ConfirmedCases')]                .set_index('Date')                [col ]            .plot(ax = ax, 
                  linewidth = lq * rank**-rank_pwr, legend=True, linestyle = '-.');

        
    for col in [c for c in cols if '_c_0.05' in c]:
        rank = score_df.loc[col.split('_c_0.')[0]]['rank']
        ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='ConfirmedCases')]                .set_index('Date')                [col ]            .plot(ax = ax, 
                  linewidth = llq * rank**-rank_pwr, legend=True, linestyle = '-.');
#     break


# ###   

# ## Fatalities

# In[ ]:


for place in places:
    
    
    ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='Fatalities')]            .set_index('Date')            [["TargetValue"] ]        .plot(title = '{} - Fatalities'.format(place), 
              linewidth = lt, legend=True);
    
    
    for col in [c for c in cols if '_f_0.5' in c]:
        rank = score_df.loc[col.split('_f_0.')[0]]['rank']
        ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='Fatalities')]                .set_index('Date')                [col ]            .plot(ax = ax, 
                  linewidth = lm * rank**-rank_pwr, legend=True);

    for col in [c for c in cols if '_f_0.95' in c]:
        rank = score_df.loc[col.split('_f_0.')[0]]['rank']
        ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='Fatalities')]                .set_index('Date')                [col]            .plot( ax = ax, 
                  linewidth = lq * rank**-rank_pwr, legend=True, linestyle = '-.');

        
    for col in [c for c in cols if '_f_0.05' in c]:
        rank = score_df.loc[col.split('_f_0.')[0]]['rank']
        ax = matched_final[(matched_final.Place==place) & (matched_final.Target=='Fatalities')]                .set_index('Date')                [col]            .plot( ax = ax, 
                  linewidth = llq * rank**-rank_pwr, legend=True, linestyle = '-.');
#     break


# ###   

# ###   
