#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


dir_ihme = '/kaggle/input/ihme-projections-for-covid19'
dir_ks = '/kaggle/input/covid19-global-forecasting-submissions'
results_file = '/kaggle/input/covid19-global-forecasting-week-4/train.csv'
test_file = results_file.replace('train', 'test').replace('-4', '-#')


# In[ ]:


ihme_list = os.listdir(dir_ihme)
list.sort(ihme_list)
print(ihme_list)

ks_list = os.listdir(dir_ks)
list.sort(ks_list)
print(ks_list)

test_list = [ test_file.replace('#', str(n)) for n in [1, 2, 3, 4] ]
print(test_list)

cutoff_dates = ['2020-03-25', '2020-04-01', '2020-04-08', '2020-04-15']
print(cutoff_dates)


# In[ ]:





# In[ ]:


def RMSLE(df, verbose = 0):
    def score(scores):
        return np.sqrt(np.sum( np.power(scores, 2) ) / len(scores))
    
    all_scores = []
    for col in ['Fatalities']:
        scores = np.log(df[col + '_true'] + 1) - np.log(df[col + '_pred']+ 1)
        df[col[:1] + '_error'] = scores
        
#         print("{}: {:.2f}".format(col, score(scores)))
        all_scores.append(score(scores))
        
#         print("   pos: {:.2f}".format( score (np.clip(scores, 0, None) ) ))  
#         print("   neg: {:.2f}".format( score (np.clip(scores, None, 0) ) ))
        
        if verbose > 1:
            print(np.round(scores.reset_index().drop(columns='Place').groupby('Date').
                      apply(score),2))
#        print(scores.reset_index().groupby('Date').apply(score))
 #       print(scores.groupby(df.reset_index().Date).apply(score))
            print()
    
#     print()
#     print("Overall Score: {:.2f}".format(np.mean(all_scores)))
    return np.mean(all_scores), df
        

kaggles
# In[ ]:


def get_IHME_merges(r, ihme, date_col, cutoff_date):
    ihme_states = pd.merge(r[r.Country_Region == 'US'], 
                           ihme[['location_name', date_col, 'Fatalities']], 
                     left_on=['Province_State', 'Date'], 
                     right_on=['location_name', date_col],
                          suffixes = ('_true', '_pred'))
    ihme_countries = pd.merge(r[r.Province_State.isnull()], 
                              ihme[['location_name', date_col, 'Fatalities']], 
                     left_on=['Country_Region', 'Date'], 
                     right_on=['location_name', date_col],
                             suffixes = ('_true', '_pred'))
    ihme_countries = ihme_countries[~ihme_countries.location_name.isin(ihme_states.location_name)]
    
    ihme_states = ihme_states[ihme_states.Date > cutoff_date]
    ihme_countries = ihme_countries[ihme_countries.Date > cutoff_date]
    
    
    
    return ihme_states, ihme_countries


# In[ ]:


def get_K_merges(r, ks, ihme_states, ihme_countries, cutoff_date):
    k_states = pd.merge(r[(r.Country_Region == 'US')
                                     & (r.Province_State.isin(ihme_states.Province_State))], ks, 
                                on = ['Country_Region', 'Province_State', 'Date'],
                            suffixes = ('_true', '_pred'))
    k_countries = pd.merge(r[(r.Province_State.isnull()) 
                                     & (r.Country_Region.isin(ihme_countries.Country_Region.unique()))
                                            & (~r.Country_Region.isin(ihme_states.Province_State))], 
                               ks, 
                                on = ['Country_Region', 'Province_State', 'Date'],
                            suffixes = ('_true', '_pred'))
    
    k_states = k_states[k_states.Date > cutoff_date]
    k_countries = k_countries[k_countries.Date > cutoff_date]
    
    
    return k_states, k_countries


# In[ ]:


for idx, _ in enumerate(ks_list):
    if idx < 2:
        continue;
    cutoff_date = cutoff_dates[idx]
    
    ihme = pd.read_csv(os.path.join(dir_ihme, ihme_list[idx]))
    print("{}: {} locations".format(ihme_list[idx], len(ihme.location_name.unique())));
    date_col = [x for x in ihme.columns if 'date' in x][0] 
    ihme.rename(columns = {'totdea_mean': 'Fatalities'}, inplace=True)
    
    # find Kaggles
    kaggles = os.listdir(os.path.join(dir_ks, ks_list[idx]))
#     print(len(kaggles))
    
    # merge with actual results
    r = pd.read_csv(results_file)
    r = r[(r.Fatalities >= 0 )]
    ihme_states, ihme_countries = get_IHME_merges(r, ihme, date_col, cutoff_date)
    
    # pare down dates
    
#     print(len(ihme_states))
#     print(len(ihme_countries))
    
    
    state_rmsles = []
    test = pd.read_csv(test_list[idx])
    test.columns = [x.replace('/', '_') for x in test.columns]
    for k in kaggles:
        ks = pd.read_csv(os.path.join(dir_ks, ks_list[idx], k))
        if set(ks.columns) != set(['ForecastId', 'ConfirmedCases', 'Fatalities']):
#             print('invalid sub: columns are off')
            continue
        ks = pd.merge(ks, test, on='ForecastId')
        k_states, k_countries = get_K_merges(r, ks, ihme_states, ihme_countries, cutoff_date)
        if (set(k_states.Province_State) == set(ihme_states.Province_State))                    and len(k_states) == len(ihme_states):
            try:
                state_rmsles.append(RMSLE(k_states)[0])
            except:
                print('invalid sub: column mismatch')
#             print('valid sub')
        else:
            print('invalid sub')
    list.sort(state_rmsles)
    
    
    # Results
    print("Projections as of {}".format(cutoff_date))
    print('  scoring {} fatality counts from {} states on {} dates'.format(
                    len(ihme_states), len(ihme_states.location_name.unique()), 
                                        len(ihme_states.Date.unique())))
#     print(state_rmsles)
    r, _ = RMSLE(ihme_states)
    print()
    print("IHME Rank: {} of {}".format(sum([(r > s) for s in state_rmsles]) + 1, len(state_rmsles))) 
    
    print("   IHME RMSLE:   {:.3f}".format(r));
    print()
    print("   Kaggle Best:  {:.3f}".format(state_rmsles[0]))
    print("   Kaggle 10th:  {:.3f}".format(state_rmsles[9]))
    print("   Kaggle 50th:  {:.3f}".format(state_rmsles[49]))
    print(); print()
           

