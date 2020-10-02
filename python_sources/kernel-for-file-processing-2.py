#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import time
import warnings

#from pycountry_convert import ( map_countries, country_name_to_country_alpha3,)
import pytz as pytz
import datetime

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#lgm and graph viz
import graphviz 
import lightgbm as lgb

warnings.filterwarnings('ignore')


# In[ ]:


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
      
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str', 'visitId':'str', 'visitStartTime':'str', 'date':'str'}, 
                     nrows=nrows)

    #Normalize JSON colunmns and drop
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df


def drop_constant_cols(df):
    ## Drop constant columns
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    df.drop(const_cols, axis=1, inplace=True)
    
    #this columnm is only in train data
    try:
        df.drop('trafficSource.campaignCode', axis=1, inplace=True)   
    except:
        None   
    


# In[ ]:


os.listdir('../input')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Load\ntrain_df = load_df(csv_path='../input/ga-customer-revenue-prediction/train.csv', nrows = None)\n#train_df.to_pickle('train_flat_no_drop.pkl')\ndrop_constant_cols(train_df)\n\ntest_df = load_df(csv_path='../input/ga-customer-revenue-prediction/test.csv', nrows = None)\n#train_df.to_pickle('test_flat_no_drop.pkl')\ndrop_constant_cols(test_df)")


# In[ ]:


# Extract target values and Ids
cat_cols = ['channelGrouping','device.browser',
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent','trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source'  ]


num_cols = ['visitNumber', 'totals.bounces', 'totals.hits',
            'totals.newVisits', 'totals.pageviews', 
            '_local_hourofday'  ]

interaction_cols = ['totals.hits / totals.pageviews', 'totals.hits * totals.pageviews',
       'totals.hits - totals.pageviews']

visitStartTime = ['visitStartTime']

ID_cols = ['date', 'fullVisitorId', 'sessionId', 'visitId']

target_col = ['totals.transactionRevenue']


# In[ ]:


os.listdir('../input/geocodes-timezones')


# In[ ]:


#Load
geocode_df= pd.read_pickle('../input/geocodes-timezones/geocodes_timezones.pkl')

def time_zone_converter(x):
    
    try:
        return pytz.country_timezones(x)[0]
    except AttributeError:
        return np.nan
   

def time_localizer(s):
    #format of series [time,zone]
    try:
        tz =pytz.timezone(s[1])
        return pytz.utc.localize(s[0], is_dst=None).astimezone(tz)
    except:
        return np.nan
    
def remove_missing_vals(x):
    remove_list = ['(not set)', 'not available in demo dataset','unknown.unknown']
    if x in remove_list:
        return ''
    else:
        return x 
    
def map_timezone(x):   
    try:
        return timezone_dict[x]
    except KeyError:
        return 'UTC'


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df['visitStartTime'] = pd.to_datetime(train_df['visitStartTime'], unit = 's')\ntest_df['visitStartTime'] = pd.to_datetime(test_df['visitStartTime'], unit = 's')\n\n#Generate foreign key '_search_term' by concatenating city, region, country\ntrain_df['_search_term'] = train_df['geoNetwork.city'].map(remove_missing_vals) + ' ' + train_df['geoNetwork.region'].map(remove_missing_vals) + ' ' + train_df['geoNetwork.country'].map(remove_missing_vals)\ntest_df['_search_term'] = test_df['geoNetwork.city'].map(remove_missing_vals) + ' ' + test_df['geoNetwork.region'].map(remove_missing_vals) + ' ' + test_df['geoNetwork.country'].map(remove_missing_vals)\n\n#Set global variable, needed for map_timezone function\nglobal timezone_dict\ntimezone_dict = dict(zip(geocode_df['search_term'], geocode_df['timeZoneId']))\n\n#Map timezones\ntrain_df['_timeZoneId'] = train_df['_search_term'].map(map_timezone)\ntest_df['_timeZoneId'] = test_df['_search_term'].map(map_timezone)\n  \n#Create time zone aware column\ntrain_df['_local_time'] = train_df[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1).astype(str)\ntest_df['_local_time'] = test_df[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1).astype(str)  \n\n#Localize hour time\ntrain_df['_local_hourofday'] = train_df['_local_time'].str[11:13]\ntest_df['_local_hourofday'] = test_df['_local_time'].str[11:13]\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def map_longitude(x):   \n    try:\n        return longitude_dict[x]\n    except KeyError:\n        return np.nan\n    \ndef map_latitude(x):   \n    try:\n        return latitude_dict[x]\n    except KeyError:\n        return np.nan\n    \nglobal longitude_dict\nlongitude_dict = dict(zip(geocode_df['search_term'], geocode_df['geometry.location.lng']))\n\nglobal latitude_dict\nlatitude_dict = dict(zip(geocode_df['search_term'], geocode_df['geometry.location.lat']))\n\n\n#Map latitude\ntrain_df['_latitude'] = train_df['_search_term'].map(map_latitude)\ntest_df['_latitude'] = test_df['_search_term'].map(map_latitude)\n\n#Map longitude\ntrain_df['_longitude'] = train_df['_search_term'].map(map_longitude)\ntest_df['_longitude'] = test_df['_search_term'].map(map_longitude)")


# # Time since last visit 

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_ts = train_df[['fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime']].copy()\ntest_ts = test_df[['fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime']].copy()\n\n\ntrain_df['_time_since_last_visit'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff()\ntrain_df['_time_since_last_visit_2'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(2)\ntest_df['_time_since_last_visit'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff()\ntest_df['_time_since_last_visit_2'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(2)\n\ntrain_df['_time_to_next_visit'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-1)\ntrain_df['_time_to_next_visit_2'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-2)\ntest_df['_time_to_next_visit'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-1)\ntest_df['_time_to_next_visit_2'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-2)\n\n#del train_ts\n#del test_ts")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for col in ['totals.bounces', 'totals.hits','totals.pageviews',  '_local_hourofday']:\n    train_df['_prev_{}_1'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(1)\n    test_df['_prev_{}_1'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(1)\n    train_df['_prev_{}_2'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(2)\n    test_df['_prev_{}_2'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(2)\n    \n    train_df['_next_{}_1'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-1)\n    test_df['_next_{}_1'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-1)\n    train_df['_next_{}_2'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-2)\n    test_df['_next_{}_2'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-2)\n    ")


# ## Previous numerical values 

# test_df['_previous_'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff()
# 
# num_cols = ['visitNumber', 'totals.bounces', 'totals.hits',
#             'totals.newVisits', 'totals.pageviews', 
#             '_local_hourofday'  ]

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df['_time_first_visit'] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\\\n.transform('first')\ntrain_df['_time_last_visit'] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\\\n.transform('last')\ntrain_df['_difference_first_last'] = train_df['_time_last_visit'] - train_df['_time_first_visit']\ntrain_df['_time_since_first_visit'] = train_df['visitStartTime'] - train_df['_time_first_visit']\ntrain_df.drop(['_time_first_visit', '_time_last_visit'], axis = 1,inplace = True)\n\n\ntest_df['_time_first_visit'] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\\\n.transform('first')\ntest_df['_time_last_visit'] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\\\n.transform('last')\ntest_df['_difference_first_last'] = test_df['_time_last_visit'] - test_df['_time_first_visit']\ntest_df['_difference_first_last'] = test_df['_time_last_visit'] - test_df['_time_first_visit']\ntest_df['_time_since_first_visit'] = test_df['visitStartTime'] - test_df['_time_first_visit']\ntest_df.drop(['_time_first_visit', '_time_last_visit'], axis = 1,inplace = True)\n")


# In[ ]:


train_df.info()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#train_df['_time_since_last_visit'] = pd.to_numeric(train_df['_time_since_last_visit'])\n#test_df['_time_since_last_visit'] = pd.to_numeric(test_df['_time_since_last_visit'])\n\n#Preparation\nwip_cols = ['fullVisitorId', 'sessionId', 'visitId',\n       'visitNumber', 'visitStartTime','totals.bounces', 'totals.hits',\n       'totals.newVisits', 'totals.pageviews', '_time_since_last_visit']\n\ntrain_ts = train_df.sort_values(['fullVisitorId', 'visitStartTime']).reset_index()\ntrain_ts['index'] = train_ts['index'].astype('str')\ntrain_ts_grouped = train_ts.groupby('fullVisitorId')\n\n#Calculating rolling frequency\ntemp_roll = train_ts_grouped.rolling('12H', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_12H') \ntrain_ts = pd.concat([train_ts, temp_roll['visitNumber_12H']], axis = 1)\n\ntemp_roll = train_ts_grouped.rolling('7D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_7D') \ntrain_ts = pd.concat([train_ts, temp_roll['visitNumber_7D']], axis = 1)\n\ntemp_roll = train_ts_grouped.rolling('30D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_30D') \ntrain_ts = pd.concat([train_ts, temp_roll['visitNumber_30D']], axis = 1)\n\ntrain_ts['index'] = train_ts['index'].astype('int')\ntrain_ts.set_index('index', inplace = True)\ntrain_ts.sort_index(inplace = True)\ntrain_df = train_ts.copy()\ndel train_ts")


# In[ ]:


train_df.info()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntest_ts = test_df.sort_values(['fullVisitorId', 'visitStartTime']).reset_index()\ntest_ts['index'] = test_ts['index'].astype('str')\ntest_ts_grouped = test_ts.groupby('fullVisitorId')\n\n#Calculating rolling frequency\ntemp_roll = test_ts_grouped.rolling('12H', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_12H') \ntest_ts = pd.concat([test_ts, temp_roll['visitNumber_12H']], axis = 1)\n\ntemp_roll = test_ts_grouped.rolling('7D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_7D') \ntest_ts = pd.concat([test_ts, temp_roll['visitNumber_7D']], axis = 1)\n\ntemp_roll = test_ts_grouped.rolling('30D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_30D')\ntest_ts = pd.concat([test_ts, temp_roll['visitNumber_30D']], axis = 1)\n\ntest_ts['index'] = test_ts['index'].astype('int')\ntest_ts.set_index('index', inplace = True)\ntest_ts.sort_index(inplace = True)\ntest_df = test_ts.copy()\ndel test_ts\n")


# In[ ]:


test_df.info()


# In[ ]:





# In[ ]:





# In[ ]:


train_df.to_pickle('train_flat_FE.pkl')
test_df.to_pickle('test_flat_FE.pkl')


# # Categoricals processing 

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Categorical encoding\nfor c in cat_cols:\n    #Convert NAs to unknown\n    train_df[c] = train_df[c].fillna(\'unknown\')\n    test_df[c] = test_df[c].fillna(\'unknown\')\n\n\n#Rename "Other" those with less than 10\nfor col in cat_cols:\n    #For train data\n    series1 = pd.value_counts(train_df[col])\n    mask1 = series1 < 10\n    train_df[col] = np.where(train_df[col].isin(series1[mask1].index),\'Other_{}\'.format(col), train_df[col])\n    \n    #For test data\n    series2 = pd.value_counts(test_df[col])\n    mask2 = series2 < 10\n    test_df[col] = np.where(test_df[col].isin(series2[mask2].index),\'Other_{}\'.format(col), test_df[col])\n    ')


# In[ ]:


get_ipython().run_cell_magic('time', '', "interact_cats = ['channelGrouping', 'device.operatingSystem',\n                'geoNetwork.city', 'geoNetwork.country', 'geoNetwork.networkDomain',\n                 'trafficSource.medium', \n                'trafficSource.referralPath', 'trafficSource.source']\n\n#2-way interactions\nfrom itertools import combinations\n\ndef categorical_interaction_terms_2(df, columns):\n    for c in combinations(columns,2):\n        df['{}+{}'.format(c[0], c[1]) ] = df[c[0]] + '_' + df[c[1]]\n    return df\n\ndef categorical_interaction_terms_3(df, columns):\n    for c in combinations(columns,3):\n        df['{}+{}+{}'.format(c[0], c[1], c[2]) ] = df[c[0]] + '_' + df[c[1]] + '_' + df[c[2]]\n    return df\n\ntrain_df = categorical_interaction_terms_2(train_df,interact_cats )\n#train_df = categorical_interaction_terms_3(train_df,interact_cats )\n\ntest_df = categorical_interaction_terms_2(test_df,interact_cats )\n#test_df = categorical_interaction_terms_3(test_df,interact_cats )\n\ninteract_cats_to_keep = [ 'geoNetwork.city+geoNetwork.networkDomain',\n  'device.operatingSystem+geoNetwork.networkDomain',\n  'device.operatingSystem+geoNetwork.city', \n  'channelGrouping+geoNetwork.networkDomain',\n  'geoNetwork.city+trafficSource.source',\n 'geoNetwork.networkDomain+trafficSource.source',\n 'geoNetwork.networkDomain+trafficSource.referralPath',\n 'geoNetwork.networkDomain+trafficSource.medium',\n 'geoNetwork.city+trafficSource.medium',\n 'geoNetwork.city+geoNetwork.country']\n")


# # Label encoding 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#Factorize cats\nfor f in (cat_cols + interact_cats_to_keep ):\n    train_df[f], indexer = pd.factorize(train_df[f])\n    test_df[f] = indexer.get_indexer(test_df[f])\n\ndel indexer')


# In[ ]:


train_df.to_pickle('train_flat_FE_CAT_LE.pkl')
test_df.to_pickle('test_flat_FE_CAT_LE.pkl')


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()

