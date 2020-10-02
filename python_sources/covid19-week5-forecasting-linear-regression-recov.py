#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# This is week 5 of Kaggle's COVID19 forecasting series.
# 
# Here we also forecast each country's recovery rate. If forecasted recoveries > forecasted confirmed cases, this means the curve for forecasted confirmedcases have flattened and will be adjusted.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import math
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

from datetime import timedelta
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


N = 3 # Number of previous data points to use to forecast confirmedcases
N_ft = 5 # Number of previous data points to use to forecast fatalities
N_rc = 3 # Number of previous data points to use to forecast recoveries
z_val = 1.645 # For 90% confidence interval


# # Common functions

# In[ ]:


def get_preds_lin_reg(series, pred_min, H):
    """
    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        series     : series to forecast
        pred_min   : all predictions should be >= pred_min
        H          : forecast horizon
    Outputs
        result: the predictions. The length of result is H. numpy array of shape (H,)
    """
    # Create linear regression object
    regr = LinearRegression(fit_intercept=True)

    pred_list = []

    X_train = np.array(range(len(series))) # e.g. [0 1 2 3 4]
    y_train = np.array(series) # e.g. [2944 3088 3226 3335 3436]
    X_train = X_train.reshape(-1, 1)     # e.g X_train = 
                                             # [[0]
                                             #  [1]
                                             #  [2]
                                             #  [3]
                                             #  [4]]
    # X_train = np.c_[np.ones(N), X_train]              # add a column
    y_train = y_train.reshape(-1, 1)
    regr.fit(X_train, y_train)            # Train the model
    pred = regr.predict(np.array(range(len(series),len(series)+H)).reshape(-1,1))
    pred = pred.reshape(H,)
    
    # If the values are < pred_min, set it to be pred_min
    pred[pred < pred_min] = pred_min
        
    return np.around(pred)


# # Load data

# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

# Change column names to lower case
train.columns = [col.lower() for col in train.columns]

# Change to date format
train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')

train


# In[ ]:


test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

# Change column names to lower case
test.columns = [col.lower() for col in test.columns]

# Change to date format
test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')

test


# In[ ]:


submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
submission


# # Load recoveries data

# In[ ]:


# Get recovery data
# url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
# recov = pd.read_csv(url, error_bad_lines=False)

recov = pd.read_csv('../input/time-series-covid19-recovered-global/time_series_covid19_recovered_global.csv')
recov


# In[ ]:


# Convert recoveries to the right format
province_state = []
country_region = []
date = []
recoveries = []

dates = list(recov.columns[4:])
for index, row in recov.iterrows():
    province_state = province_state + [row['Province/State']]*len(dates)
    country_region = country_region + [row['Country/Region']]*len(dates)
    date = date + dates
    recoveries = recoveries + list(row[4:])
    
recoveries_df = pd.DataFrame({'province_state': province_state,
                              'country_region': country_region,
                              'date': date,
                              'recoveries_tot': recoveries})

# Change to date format
recoveries_df['date'] = pd.to_datetime(recoveries_df['date'], format='%m/%d/%y')

# Add a column 'county'
recoveries_df['county'] = 'nil'

# Change NaN to nil
recoveries_df['province_state'] = recoveries_df['province_state'].fillna(value = 'nil')

recoveries_df


# # EDA

# In[ ]:


# Count number of nulls for each column
train.isnull().sum(axis=0)


# In[ ]:


# Count number of nulls for each column
recoveries_df.isnull().sum(axis=0)


# In[ ]:


# Get the counties
print(len(train['county'].unique()))
print(train['county'].unique().tolist())


# In[ ]:


# Get the province_states
print(len(train['province_state'].unique()))
train['province_state'].unique()


# In[ ]:


# Get the country_regions
print(len(train['country_region'].unique()))
train['country_region'].unique()


# In[ ]:


# Get amount of data per country
train['country_region'].value_counts()


# In[ ]:


train[train['country_region']=='Singapore']


# In[ ]:


# Plot the confirmed cases (daily) and fatalities (daily) in SEA
countries_list = ['Singapore', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Brunei', 'Laos', 'Cambodia', 'New Zealand']
color_list = ['r', 'g', 'b', 'k', 'c', 'y', 'm', '0.75', 'tab:blue']

fig, axes = plt.subplots(2,1)
ax = train[(train['country_region']==countries_list[0]) & (train['target']=='ConfirmedCases') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(ax=axes[0], x='date', y='targetvalue', color=color_list[0], marker='.', grid=True, figsize=(16, 8))
ax1 = train[(train['country_region']==countries_list[0]) & (train['target']=='Fatalities') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(ax=axes[1], x='date', y='targetvalue', color=color_list[0], marker='.', grid=True)

i = 1
for country in countries_list[1:]:
    ax = train[(train['country_region']==country) & (train['target']=='ConfirmedCases') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(x='date', y='targetvalue', color=color_list[i%len(color_list)], marker='.', grid=True, ax=ax)
    ax1 = train[(train['country_region']==country) & (train['target']=='Fatalities') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(x='date', y='targetvalue', color=color_list[i%len(color_list)], marker='.', grid=True, ax=ax1)
    i = i + 1
    
ax.set_xlabel("date")
ax.set_ylabel("confirmedcases (daily)")
ax.legend(countries_list, loc=2)

ax1.set_xlabel("date")
ax1.set_ylabel("fatalities (daily)")
ax1.legend(countries_list, loc=2)


# In[ ]:


# Plot the confirmed cases (daily) and fatalities (daily) in big countries
countries_list = ['China', 'US', 'India', 'Italy', 'France', 'Iran']
color_list = ['r', 'g', 'b', 'k', 'c', 'y', 'm', '0.75', 'tab:blue']

fig, axes = plt.subplots(2,1)
ax = train[(train['country_region']==countries_list[0]) & (train['target']=='ConfirmedCases') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(ax=axes[0], x='date', y='targetvalue', color=color_list[0], marker='.', grid=True, figsize=(16, 8))
ax1 = train[(train['country_region']==countries_list[0]) & (train['target']=='Fatalities') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(ax=axes[1], x='date', y='targetvalue', color=color_list[0], marker='.', grid=True)

i = 1
for country in countries_list[1:]:
    ax = train[(train['country_region']==country) & (train['target']=='ConfirmedCases') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(x='date', y='targetvalue', color=color_list[i%len(color_list)], marker='.', grid=True, ax=ax)
    ax1 = train[(train['country_region']==country) & (train['target']=='Fatalities') & (train['county'].isnull()) & (train['province_state'].isnull())].plot(x='date', y='targetvalue', color=color_list[i%len(color_list)], marker='.', grid=True, ax=ax1)
    i = i + 1
    
ax.set_xlabel("date")
ax.set_ylabel("confirmedcases (daily)")
ax.legend(countries_list, loc=2)

ax1.set_xlabel("date")
ax1.set_ylabel("fatalities (daily)")
ax1.legend(countries_list, loc=2)


# Why are there negative values in the targetvalues?

# # Pre-process train, test

# In[ ]:


# Fill nans in province_state and county with ''
train['province_state'] = train['province_state'].fillna(value = 'nil')
train['county'] = train['county'].fillna(value = 'nil')
test['province_state'] = test['province_state'].fillna(value = 'nil')
test['county'] = test['county'].fillna(value = 'nil')


# In[ ]:


# Get unique combinations of province_state and country_region
ct_ps_cr_unique = train[['county', 'province_state', 'country_region']].drop_duplicates()
ct_ps_cr_unique


# In[ ]:


# Get number of days we need to predict
date_max_train = train[(train['province_state']=='nil') & 
                       (train['county']=='nil') & 
                       (train['country_region']=='Singapore')]['date'].max()

date_max_test = test[(test['province_state']=='nil') &
                     (test['county']=='nil') &
                     (test['country_region']=='Singapore')]['date'].max()

pred_days = (date_max_test - date_max_train).days
print(date_max_train, date_max_test, pred_days)


# In[ ]:


# Split train set
train_cc = train[train['target']=='ConfirmedCases']
train_ft = train[train['target']=='Fatalities']
train_cc


# In[ ]:


# Do cumsum to get total cases
tic = time.time()
train_cc_tot = train_cc[(train_cc['county']==ct_ps_cr_unique.iloc[0]['county']) & 
                          (train_cc['province_state']==ct_ps_cr_unique.iloc[0]['province_state']) & 
                          (train_cc['country_region']==ct_ps_cr_unique.iloc[0]['country_region'])].copy()
train_cc_tot.loc[:, 'targetvalue_tot'] = train_cc_tot['targetvalue'].cumsum()

train_ft_tot = train_ft[(train_ft['county']==ct_ps_cr_unique.iloc[0]['county']) & 
                          (train_ft['province_state']==ct_ps_cr_unique.iloc[0]['province_state']) & 
                          (train_ft['country_region']==ct_ps_cr_unique.iloc[0]['country_region'])].copy()
train_ft_tot.loc[:, 'targetvalue_tot'] = train_ft_tot['targetvalue'].cumsum()

for index, row in ct_ps_cr_unique[1:].iterrows():
    train_cc_temp = train_cc[(train_cc['county']==row['county']) & 
                          (train_cc['province_state']==row['province_state']) & 
                          (train_cc['country_region']==row['country_region'])].copy()
    train_cc_temp.loc[:, 'targetvalue_tot'] = train_cc_temp['targetvalue'].cumsum()
    train_cc_tot = pd.concat([train_cc_tot, train_cc_temp], axis=0)
    
    train_ft_temp = train_ft[(train_ft['county']==row['county']) & 
                          (train_ft['province_state']==row['province_state']) & 
                          (train_ft['country_region']==row['country_region'])].copy()
    train_ft_temp.loc[:, 'targetvalue_tot'] = train_ft_temp['targetvalue'].cumsum()
    train_ft_tot = pd.concat([train_ft_tot, train_ft_temp], axis=0)

toc = time.time()
print("Time taken = " + str((toc-tic)/60.0) + " mins")
train_cc_tot


# In[ ]:


# Merge recoveries_df with train
train_cc_tot_merged = train_cc_tot.merge(recoveries_df,
                           left_on=['county', 'province_state', 'country_region', 'date'], 
                           right_on=['county', 'province_state', 'country_region', 'date'], 
                           how='left')
train_cc_tot_merged


# In[ ]:


# Count number of nulls for each column
train_cc_tot_merged.isnull().sum(axis=0)


# In[ ]:


# Fill recoveries nans with 0
train_cc_tot_merged['recoveries_tot'] = train_cc_tot_merged['recoveries_tot'].fillna(value = 0)


# # Prediction for one country

# In[ ]:


# # Specify the country here
# ct = 'nil'
# ps = 'nil'
# cr = 'Singapore'


# In[ ]:


# train_sgp = train_cc_tot[(train_cc_tot['county']==ct) & (train_cc_tot['province_state']==ps) & (train_cc_tot['country_region']==cr)]
# train_sgp[-5:]


# In[ ]:


# # Get predictions 
# preds = get_preds_lin_reg(train_sgp['targetvalue_tot'][-N:], 0, pred_days)
# preds


# In[ ]:


# # Put into dataframe
# date_list = []
# date = pd.date_range(date_max_train+timedelta(days=1), date_max_test)
# results = pd.DataFrame({'date': date, 'preds':preds})
# results.head()


# In[ ]:


# # Plot the confirmed cases in Singapore and the predictions
# ax = train_cc_tot[(train_cc_tot['county']==ct) & (train_cc_tot['province_state']==ps) & (train_cc_tot['country_region']==cr)].plot(x='date', y='targetvalue_tot', style = 'r.-', grid=True, figsize=(10, 6))
# ax = results.plot(x='date', y='preds', style = 'r.', grid=True, figsize=(10, 6), ax=ax)
    

# ax.set_xlabel("date")
# ax.set_ylabel("confirmed cases total")
# ax.legend([cr])


# # Predictions for confirmedcases, fatalities, recoveries

# In[ ]:


ct_list = []
ps_list = []
cr_list = []
date_list = []
confirmedcasespred_list = []
fatalities_list = []
recoveries_list = []

tic = time.time()
for index, row in ct_ps_cr_unique.iterrows():
    train_cc_temp = train_cc_tot[(train_cc_tot['county']==row['county']) &
                                 (train_cc_tot['province_state']==row['province_state']) & 
                                 (train_cc_tot['country_region']==row['country_region'])]
    preds = get_preds_lin_reg(train_cc_temp['targetvalue_tot'][-N:], 0, pred_days)
    confirmedcasespred_list = confirmedcasespred_list + list(preds)
    
    train_ft_temp = train_ft_tot[(train_ft_tot['county']==row['county']) &
                                 (train_ft_tot['province_state']==row['province_state']) & 
                                 (train_ft_tot['country_region']==row['country_region'])]
    preds = get_preds_lin_reg(train_ft_temp['targetvalue_tot'][-N_ft:], 0, pred_days)
    fatalities_list = fatalities_list + list(preds)
    
    train_cc_temp = train_cc_tot_merged[(train_cc_tot_merged['county']==row['county']) &
                                        (train_cc_tot_merged['province_state']==row['province_state']) & 
                                        (train_cc_tot_merged['country_region']==row['country_region'])]
    preds = get_preds_lin_reg(train_cc_temp['targetvalue_tot'][-N_rc:], 0, pred_days)
    recoveries_list = recoveries_list + list(preds)
    
    ct_list = ct_list + ([row['county']]*pred_days)
    ps_list = ps_list + ([row['province_state']]*pred_days)
    cr_list = cr_list + ([row['country_region']]*pred_days)
    date_list = date_list + list(pd.date_range(date_max_train+timedelta(days=1), date_max_test).strftime("%Y-%m-%d"))
    

results = pd.DataFrame({'county': ct_list,
                        'province_state': ps_list,
                        'country_region': cr_list,
                        'date': date_list,
                        'confirmedcases_tot': confirmedcasespred_list, 
                        'fatalities_tot': fatalities_list,
                        'recoveries_tot': recoveries_list})
results['date'] = pd.to_datetime(results['date'], format='%Y-%m-%d')

toc = time.time()
print("Time taken = " + str((toc-tic)/60.0) + " mins")
results


# In[ ]:


# Get rows where recoveries > confirmedcases
x = results[results['recoveries_tot'] > results['confirmedcases_tot']]
print(x.to_string())


# In[ ]:


# For each country, if recoveries > confirmedcases, confirmedcases should stop increasing.
# e.g.
# province          country     date        confirmedcases     fatalities      recoveries
# South Australia	Australia	2020-04-20	439.0              4.0             425.0
# South Australia	Australia	2020-04-21	440.0              4.0             460.0
# South Australia	Australia	2020-04-22	441.0              4.0             494.0
# should become
# province          country     date        confirmedcases     fatalities      recoveries
# South Australia	Australia	2020-04-20	439.0              4.0             425.0
# South Australia	Australia	2020-04-21	440.0              4.0             460.0
# South Australia	Australia	2020-04-22	440.0              4.0             494.0       # here confirmedcases stopped increasing
def confirmedcases_stop(df):
    # Check if any rows where recoveries > confirmedcases
    if len(df[df['recoveries_tot']>df['confirmedcases_tot']])==0:
        return df
    else:
        # Extract the confirmedcases at the date where recoveries > confirmedcases
        confirmedcases_sat = df[df['recoveries_tot']>df['confirmedcases_tot']]['confirmedcases_tot'].iloc[0]
        
        # For all rows where recoveries > confirmedcases, set confirmedcases = confirmedcases_max
        df.loc[df['recoveries_tot']>df['confirmedcases_tot'], 'confirmedcases_tot'] = confirmedcases_sat
        
        return df

temp = results[(results['county']==ct_ps_cr_unique.iloc[0]['county']) & 
               (results['province_state']==ct_ps_cr_unique.iloc[0]['province_state']) & 
               (results['country_region']==ct_ps_cr_unique.iloc[0]['country_region'])].copy()
results_sat = confirmedcases_stop(temp)
    
tic = time.time()
for index, row in ct_ps_cr_unique[1:].iterrows():
    temp = results[(results['county']==row['county']) & 
                   (results['province_state']==row['province_state']) & 
                   (results['country_region']==row['country_region'])].copy()
    ps_cr_df = confirmedcases_stop(temp)
    results_sat = pd.concat([results_sat, ps_cr_df], axis=0) 
toc = time.time()
print("Time taken = " + str((toc-tic)/60.0) + " mins")
    
results_sat


# In[ ]:


train_cc_tot


# In[ ]:


# Get daily ConfirmedCases, Fatalities
results_sat_daily = pd.DataFrame()

for index, row in ct_ps_cr_unique.iterrows():
    temp = results_sat[(results_sat['county']==row['county']) & 
                   (results_sat['province_state']==row['province_state']) & 
                   (results_sat['country_region']==row['country_region'])].copy()
    
    temp['ConfirmedCases'] = temp['confirmedcases_tot'].diff()
    temp['Fatalities'] = temp['fatalities_tot'].diff()
    
    # Get first value for ConfirmedCases
    train_cc_temp = train_cc_tot[(train_cc_tot['county']==row['county']) &
                                 (train_cc_tot['province_state']==row['province_state']) & 
                                 (train_cc_tot['country_region']==row['country_region'])]    
    temp.loc[temp.index[0], 'ConfirmedCases'] = temp.loc[temp.index[0], 'confirmedcases_tot'] -                                                 train_cc_temp.loc[train_cc_temp.index[-1], 'targetvalue_tot']
    
    # Get first value for Fatalities
    train_ft_temp = train_ft_tot[(train_ft_tot['county']==row['county']) &
                                 (train_ft_tot['province_state']==row['province_state']) & 
                                 (train_ft_tot['country_region']==row['country_region'])]    
    temp.loc[temp.index[0], 'Fatalities'] = temp.loc[temp.index[0], 'fatalities_tot'] -                                                 train_ft_temp.loc[train_ft_temp.index[-1], 'targetvalue_tot']
    
    results_sat_daily = results_sat_daily.append(temp)
    
results_sat_daily


# # Get std dev of the predictions for confirmedcases, fatalities, recoveries

# In[ ]:


# Use 20% of train_cc_tot to calculate the std dev of the pred error
date_max_train = train[(train['province_state']=='nil') & 
                       (train['county']=='nil') & 
                       (train['country_region']=='Singapore')]['date'].max()

date_min_train = train[(train['province_state']=='nil') &
                     (train['county']=='nil') &
                     (train['country_region']=='Singapore')]['date'].min()

diff_days = (date_max_train - date_min_train).days
train_size = int(0.8*diff_days)
val_size = diff_days - train_size
train_size, val_size


# In[ ]:


def saturate(cc, rc):
    """
    If rc > cc, then cc should saturate
    e.g.
    cc = np.array([1, 2, 3])
    rc = np.array([1, 3, 4])
    Return np.array([1, 2, 2])
    """
    if sum(rc>cc)==0:
        return cc
    else:
        sat_value = rc[np.argmax((rc>cc)==True)]
        cc[rc>cc] = sat_value
        return cc
    


# In[ ]:


ct_list = []
ps_list = []
cr_list = []
cc_stddev_list = []
ft_stddev_list = []

tic = time.time()
for index, row in ct_ps_cr_unique.iterrows():
    # Predict confirmedcases
    train_cc_tr = train_cc_tot[(train_cc_tot['county']==row['county']) &
                               (train_cc_tot['province_state']==row['province_state']) & 
                               (train_cc_tot['country_region']==row['country_region'])][:train_size]
    train_cc_val = train_cc_tot[(train_cc_tot['county']==row['county']) &
                                (train_cc_tot['province_state']==row['province_state']) & 
                                (train_cc_tot['country_region']==row['country_region'])][train_size:]
    preds = get_preds_lin_reg(train_cc_tr['targetvalue_tot'][-N:], 0, len(train_cc_val))
    
    # Predict recoveries
    train_cc_tr = train_cc_tot_merged[(train_cc_tot_merged['county']==row['county']) &
                               (train_cc_tot_merged['province_state']==row['province_state']) & 
                               (train_cc_tot_merged['country_region']==row['country_region'])][:train_size]
    preds_recov = get_preds_lin_reg(train_cc_tr['recoveries_tot'][-N_rc:], 0, len(train_cc_val))
    preds = saturate(preds, preds_recov)
    cc_stddev_list = cc_stddev_list + [np.std(preds - train_cc_val['targetvalue_tot'])]

    # Predict fatalities
    train_ft_tr = train_ft_tot[(train_ft_tot['county']==row['county']) &
                               (train_ft_tot['province_state']==row['province_state']) & 
                               (train_ft_tot['country_region']==row['country_region'])][:train_size]
    train_ft_val = train_ft_tot[(train_ft_tot['county']==row['county']) &
                                (train_ft_tot['province_state']==row['province_state']) & 
                                (train_ft_tot['country_region']==row['country_region'])][train_size:]
    preds = get_preds_lin_reg(train_ft_tr['targetvalue_tot'][-N_ft:], 0, len(train_ft_val))
    ft_stddev_list = ft_stddev_list + [np.std(preds - train_ft_val['targetvalue_tot'])]

    ct_list = ct_list + [row['county']]
    ps_list = ps_list + [row['province_state']]
    cr_list = cr_list + [row['country_region']]
    

results = pd.DataFrame({'county': ct_list,
                        'province_state': ps_list,
                        'country_region': cr_list,
                        'cc_stddev': cc_stddev_list, 
                        'ft_stddev': ft_stddev_list})
toc = time.time()
print("Time taken = " + str((toc-tic)/60.0) + " mins")
results


# # Get 90% confidence intervals of the predictions

# In[ ]:


results_sat_daily_ci = pd.DataFrame()

tic = time.time()
for index, row in ct_ps_cr_unique.iterrows():
    temp = results_sat_daily[(results_sat_daily['county']==row['county']) & 
                             (results_sat_daily['province_state']==row['province_state']) & 
                             (results_sat_daily['country_region']==row['country_region'])].copy()
    
    cc_stddev = results[(results['county']==row['county']) & 
                        (results['province_state']==row['province_state']) & 
                        (results['country_region']==row['country_region'])]['cc_stddev']
    
    ft_stddev = results[(results['county']==row['county']) & 
                        (results['province_state']==row['province_state']) & 
                        (results['country_region']==row['country_region'])]['ft_stddev']
    
    results_sat_daily_ci = pd.concat([results_sat_daily_ci, temp], axis=0) 

toc = time.time()
print("Time taken = " + str((toc-tic)/60.0) + " mins")
results_sat_daily_ci


# # Prepare submission file

# In[ ]:


# Melt the dataframe from a wide dataframe to a long dataframe
results_sat_daily_ci_melt = pd.melt(results_sat_daily_ci, 
                                    id_vars=['county', 'province_state', 'country_region', 'date'], 
                                    value_vars=['ConfirmedCases', 'Fatalities'])
results_sat_daily_ci_melt.sort_values(['country_region', 'date'], inplace=True)
results_sat_daily_ci_melt


# In[ ]:


# Merge test with results
test_merged = test.merge(results_sat_daily_ci_melt,
                           left_on=['county', 'province_state', 'country_region', 'date', 'target'], 
                           right_on=['county', 'province_state', 'country_region', 'date', 'variable'], 
                           how='left')
test_merged.drop(['variable', 'population', 'weight'], axis=1, inplace=True)
test_merged


# In[ ]:


# Merge test with train
test_merged2 = test_merged.merge(train,
                           left_on=['county', 'province_state', 'country_region', 'date', 'target'], 
                           right_on=['county', 'province_state', 'country_region', 'date', 'target'], 
                           how='left')
test_merged2.drop(['id', 'population', 'weight'], axis=1, inplace=True)
test_merged2


# In[ ]:


# Create column TargetValue
test_merged2['TargetVal'] = test_merged2.apply(lambda row: row['targetvalue'] if pd.isnull(row['value']) else row['value'], axis=1)
test_merged2.drop(['value', 'targetvalue'], axis=1, inplace=True)
test_merged2


# In[ ]:


# Get 90% CI
test_merged2_ci = pd.DataFrame()

tic = time.time()
for index, row in ct_ps_cr_unique.iterrows():
    temp_cc = test_merged2[(test_merged2['county']==row['county']) & 
                             (test_merged2['province_state']==row['province_state']) & 
                             (test_merged2['country_region']==row['country_region']) & 
                             (test_merged2['target']=='ConfirmedCases')].copy()
    
    temp_ft = test_merged2[(test_merged2['county']==row['county']) & 
                             (test_merged2['province_state']==row['province_state']) & 
                             (test_merged2['country_region']==row['country_region']) & 
                             (test_merged2['target']=='Fatalities')].copy()
    
    cc_stddev = results[(results['county']==row['county']) & 
                        (results['province_state']==row['province_state']) & 
                        (results['country_region']==row['country_region'])]['cc_stddev']
    
    ft_stddev = results[(results['county']==row['county']) & 
                        (results['province_state']==row['province_state']) & 
                        (results['country_region']==row['country_region'])]['ft_stddev']
    
    temp_cc['low'] = (temp_cc['TargetVal'] - (z_val*cc_stddev.values[0]/math.sqrt(val_size))).astype(int)
    temp_cc['low'] = temp_cc['low'].apply(lambda row: max(0, row))
    
    temp_cc['high'] = (temp_cc['TargetVal'] + (z_val*cc_stddev.values[0]/math.sqrt(val_size))).astype(int)
    
    temp_ft['low'] = (temp_ft['TargetVal'] - (z_val*ft_stddev.values[0]/math.sqrt(val_size))).astype(int)
    temp_ft['low'] = temp_ft['low'].apply(lambda row: max(0, row))
    
    temp_ft['high'] = (temp_ft['TargetVal'] + (z_val*ft_stddev.values[0]/math.sqrt(val_size))).astype(int)
    
    temp = pd.concat([temp_cc, temp_ft], axis=0)
    temp.sort_values(['date', 'target'], inplace=True)
    
    test_merged2_ci = pd.concat([test_merged2_ci, temp], axis=0) 

toc = time.time()
print("Time taken = " + str((toc-tic)/60.0) + " mins")
test_merged2_ci


# In[ ]:


# Create ForecastID_Quantile column
test_merged2_ci['ForecastID_Quantile'] = test_merged2_ci.apply(lambda row: [str(row['forecastid'])+'_0.05', str(row['forecastid'])+'_0.5', str(row['forecastid'])+'_0.95'], axis=1)
test_merged2_ci['TargetValue'] = test_merged2_ci.apply(lambda row: [row['low'], row['TargetVal'], row['high']], axis=1)
test_merged2_ci


# In[ ]:


# Explode 
submission = test_merged2_ci[['ForecastID_Quantile', 'TargetValue']].apply(pd.Series.explode)
submission


# In[ ]:


# Test submission
submission.to_csv("submission.csv", index=False)

