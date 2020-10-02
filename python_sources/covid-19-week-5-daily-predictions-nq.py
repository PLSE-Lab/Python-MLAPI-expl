#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Daily Counts Model (Northquay)

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


# ### Load

# In[ ]:


stack_path = '/kaggle/input/tempdata'
stacks = []
files = os.listdir(stack_path)
files = [file for file in files if 'c19' in file]
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

train['Place'] = train.Country + ('_' + train.Province_State).fillna("") +  ('_' + train.County ).fillna("")
test['Place'] = test.Country +  ('_' + test.Province_State).fillna("") + ('_' + test.County).fillna("") 

train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)

train_bk = train.copy()
full_train = train.copy()


# ### Join with Test

# In[ ]:


pdt = ['Place', 'Date', 'Target']

stack['ForecastId'] = stack.index.str.split('_').str[0].astype(int)
stack['Quantile'] = stack.index.str.split('_').str[1]


# In[ ]:


MIN_DATE = test.Date.min() # datetime.datetime(2020, 3, 1)


# In[ ]:


full_stack = stack.merge(test, on='ForecastId', validate='m:1')    .merge(train, on=pdt, suffixes=('', '_y'), how='outer')
full_stack = full_stack[full_stack.Date >= MIN_DATE]

non_zero = full_stack[np.percentile(full_stack.iloc[:,:N_STACKS], 20, axis = 1) != 0]

non_zero.iloc[:,:N_STACKS].corr()


# ### FEATURES

# In[ ]:


full_stack['elapsed'] = (full_stack.Date - full_stack.Date.min()).dt.days +1

preds = full_stack.iloc[:, :N_STACKS]

full_stack['mean_pred'] = np.mean(preds, axis = 1)
full_stack['stdev_pred'] = np.std(preds, axis = 1)
full_stack['mean_over_stdev_pred'] = (full_stack.mean_pred / full_stack.stdev_pred ).fillna(-1)

full_stack['mean_log_pred'] = np.mean( np.log( 1 + np.clip(preds, 0, None)), axis = 1)
full_stack['stdev_log_pred'] = np.std( np.log( 1 + np.clip(preds, 0, None)), axis = 1)
full_stack['mean_over_stdev_log_pred'] = ( full_stack.mean_log_pred / full_stack.stdev_log_pred ).fillna(-1)


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


UNIQUE_COLS = full_stack_clean.columns[full_stack_clean.iloc[:1000,:].groupby(['Place', 'Date']).nunique().mean() > 1]


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
                                     for c in stack.iloc[:, :N_STACKS].columns]].iloc[:,-1] 


# In[ ]:


if len(test) > 0:
    base_date = test.Date.min() - datetime.timedelta(1)
else:
    base_date = train.Date.max()


# ### Get Predictions Merged with Test/Train Data

# In[ ]:


test_nq = pd.merge(test, data_nq[['Date', 'Place', 'ConfirmedCases_pred', 'Fatalities_pred',
                                      'ConfirmedCases5_pred', 'Fatalities5_pred',
                                      'ConfirmedCases95_pred', 'Fatalities95_pred',
                                    'elapsed']], 
            how='outer', on = ['Date', 'Place'])
test_nq = test_nq[test_nq.Date >= MIN_DATE]


# In[ ]:


cs = ['ConfirmedCases', 'Fatalities']


# In[ ]:


matched_nq = pd.merge(test_nq[pdt + [c for c in test_nq if any(z in c for z in cs)]].dropna(),
             train_bk,
                    on = pdt, how='outer')
matched_nq = matched_nq[matched_nq.Date >= MIN_DATE]


# In[ ]:


matched_nq.loc[~matched_nq.TargetValue.isnull(), 
               ['ConfirmedCases_pred',  'ConfirmedCases5_pred','ConfirmedCases95_pred',
                    'Fatalities_pred','Fatalities5_pred','Fatalities95_pred']] = np.nan


# In[ ]:


for col in ['ConfirmedCases_pred',  'ConfirmedCases5_pred','ConfirmedCases95_pred']:
    matched_nq.loc[(matched_nq.Target=='ConfirmedCases') & 
                    (~matched_nq.TargetValue.isnull()),col] = \
            matched_nq.loc[(matched_nq.Target=='ConfirmedCases') & 
                          (~matched_nq.TargetValue.isnull())].TargetValue
    matched_nq[col].fillna(method='ffill', inplace=True)
for col in ['Fatalities_pred','Fatalities5_pred','Fatalities95_pred','TargetValue']:
    matched_nq.loc[(matched_nq.Target=='Fatalities') & 
                    (~matched_nq.TargetValue.isnull()),col] = \
            matched_nq.loc[(matched_nq.Target=='Fatalities') & 
                          (~matched_nq.TargetValue.isnull())].TargetValue
    matched_nq[col].fillna(method='bfill', inplace=True)

matched_nq
# In[ ]:


places = (data_cc.ewm(span = 20).mean().iloc[-1,:]    + 10 * data_fs.ewm(span = 20).mean().iloc[-1,:] ).sort_values(ascending=False).index[:30]


# In[ ]:


matched_final = matched_nq


# In[ ]:


plt.rcParams.update({'figure.max_open_warning': 0})


# ###  

# ###   

# ## Worldwide Rollup
matched_nq
# In[ ]:


matched_nq[(matched_nq.Place.str.split('_').apply(len) == 1) ].groupby(['Date']).sum().iloc[:,:6].plot(
                title = 'Worldwide Roll-up');


# ##  

# ## US: Rollups

# In[ ]:


matched_nq[(matched_nq.Place == 'US')  &
          (matched_nq.Target=='Fatalities')].groupby(['Date']).sum().iloc[:,:6].plot(
                title='US Cases and Fatalities');


# In[ ]:


matched_nq[(matched_nq.Place != 'US') & 
               (matched_nq.Place.str.slice(0,2)=='US') &
           (matched_nq.Place.str.split('_').apply(len) == 2) &
          (matched_nq.Target=='Fatalities')].groupby(['Date']).sum().iloc[:,:6].plot(
                    title='US Cases and Fatalities - Rolled-up from States');


# In[ ]:


matched_nq[(matched_nq.Place != 'US') & 
               (matched_nq.Place.str.slice(0,2)=='US') &
           (matched_nq.Place.str.split('_').apply(len) == 3) &
          (matched_nq.Target=='Fatalities')].groupby(['Date']).sum().iloc[:,:6].plot(
                title = 'US Cases and Fatalities - Rolled-up from Counties');


# ###   

# ## Cases

# In[ ]:


for place in places:
    matched_final[(matched_final.Place==place) & (matched_final.Target=='ConfirmedCases')]            .set_index('Date')            [['ConfirmedCases_pred',  'ConfirmedCases5_pred','ConfirmedCases95_pred' ,'TargetValue']]        .plot(title = '{} - Confirmed Cases'.format(place));
        


# ###   

# ## Fatalities

# In[ ]:


for place in places:
    matched_final[(matched_final.Place==place) & (matched_final.Target=='Fatalities')]            .set_index('Date')[['Fatalities_pred','Fatalities5_pred','Fatalities95_pred','TargetValue']]    .plot(title = '{} - Fatalities'.format(place));
        


# ###   

# ###   

# In[ ]:




