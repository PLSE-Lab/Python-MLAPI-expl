#!/usr/bin/env python
# coding: utf-8

# *THIS NOTEBOOK IS SECOND IN SERIES OF THREE NOTEBOOKS *
# *     *FIRST ONE IS FOR DATA VISUALIZATION *
# *     ### **SECOND ONE IS FOR FEATURE GENERATION **
# *     *THIRD ONE IS FOR MODELLING *
#     
# 
#  

# # LTFS Data Science FinHack 2
# LTFS receives a lot of requests for its various finance offerings that include housing loan, two-wheeler loan, real estate financing and micro loans. The number of applications received is something that varies a lot with season. Going through these applications is a manual process and is tedious. Accurately forecasting the number of cases received can help with resource and manpower management resulting into quick response on applications and more efficient processing.
#     
#     
#  # Problem Statement
#  
#  You have been appointed with the task of forecasting daily cases for next 3 months for 2 different business segments aggregated at the country level keeping in consideration the following major Indian festivals (inclusive but not exhaustive list): Diwali, Dussehra, Ganesh Chaturthi, Navratri, Holi etc. (You are free to use any publicly available open source external datasets). Some other examples could be:
#      Weather Macroeconomic variables Note that the external dataset must belong to a reliable source.
#     Data Dictionary The train data has been provided in the following way:
#     For business segment 1, historical data has been made available at branch ID level For business segment 2, historical data has been made available at State level.
#     Train File Variable Definition application_date Date of application segment Business Segment (1/2) branch_id Anonymised id for branch at which application was received state State in which application was received (Karnataka, MP etc.) zone Zone of state in which application was received (Central, East etc.) case_count (Target) Number of cases/applications received
#     Test File Forecasting needs to be done at country level for the dates provided in test set for each segment.
#     Variable Definition id Unique id for each sample in test set application_date Date of application segment Business Segment (1/2)
#     
#     
# # Evaluation
# 
#  Evaluation Metric The evaluation metric for scoring the forecasts is *MAPE (Mean Absolute Percentage Error) M with the formula:
#  Where At is the actual value and Ft is the forecast value.
#  The Final score is calculated using MAPE for both the segments using the formula
#     
# 

# [https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/](http://)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/analytics-vidhya-ltfs-2/sample_submission.csv", parse_dates=['application_date'])
train = pd.read_csv("/kaggle/input/analytics-vidhya-ltfs-2/train.csv",parse_dates=['application_date'])
test = pd.read_csv("/kaggle/input/ltfs-2/test_1eLl9Yf.csv", parse_dates=['application_date'])


# In[ ]:


train_1=train[train['segment']==1].groupby(['application_date']).sum().reset_index()[['application_date','case_count']].sort_values('application_date').set_index('application_date')
train_2=train[train['segment']==2].groupby(['application_date']).sum().reset_index()[['application_date','case_count']].sort_values('application_date').set_index('application_date')
test_1=test[test['segment']==1][['application_date']].sort_values('application_date').set_index('application_date')
test_2=test[test['segment']==2][['application_date']].sort_values('application_date').set_index('application_date')


# In[ ]:


train_1.loc[train_1.case_count>=8700,'case_count']=np.nan


# In[ ]:


type(train_1.index)


# In[ ]:


train_1.index


# In[ ]:


# setting 'application_date' as column 

train_1['application_date'] = train_1.index.get_level_values('application_date') 

train_2['application_date'] = train_2.index.get_level_values('application_date') 


# setting 'application_date' as column 

test_1['application_date'] = test_1.index.get_level_values('application_date') 

test_2['application_date'] = test_2.index.get_level_values('application_date')


# In[ ]:


train_1.columns, test_1.columns


# # FEATURE GENERATION

# ### WE CAN CREATE FOLLOWING TYPE OF FETURES ON TIME SERIES DATA
#     1. TIME BASED FEATURES
#         *USING THE TIME OF DATETIME COLUMN*
#     2. DATE BASED FEATURES
#         *USING THE DATE OF DATETIME COLUMN*
#     3. TIME LAG BASED FEATURES
#         *BECAUSE TIME SERIES DATA DEPENDS ON PREVIOUS VALUES *
#     4. ROLLING MEAN FEATURES
#         *BASED ON VALUES ROLLING AT FIXED INTERVAL*
#     5. WEIGHTED MEAN FEATURES
#         *BASED ON WEIGHT DECAYS*

# In[ ]:


#WE WILL CREATE FEATURES FROM "APPLICATION_DATE" COLUMN

# THESE ARE THE STANDARDS FEATURES THAT WE ALWAYS MAKE IN CASE OF DATE COLUMN.

def create_features(df, label=None,seg=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.application_date
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['week'] = df['date'].dt.week
    df['is_month_start']=df['date'].dt.is_month_start
    df['is_month_end']=df['date'].dt.is_month_end
    df['is_quarter_start']=df['date'].dt.is_quarter_start
    df['is_quarter_end']=df['date'].dt.is_quarter_end
    df['is_year_start']=df['date'].dt.is_year_start
    df['is_year_end']=df['date'].dt.is_year_end
    df['Semester'] = np.where(df['quarter'].isin([1,2]),1,2)
    
    X = df.drop(['date','application_date'],axis=1)
    if label:
        y = df[label]
        return X
    return X

Xtrain_1 = create_features(train_1, label='case_count')

Xtest_1= create_features(test_1)


# In[ ]:


Xtrain_2 = create_features(train_2, label='case_count')

Xtest_2= create_features(test_2 )


# In[ ]:


train_1['date'] = train_1.application_date


# In[ ]:


Xtrain_1.head()
print("Shape of DataFrame: {}".format(Xtrain_1.shape))

#'Train shape: {}'.format(df.loc[df.train_or_test=='train',:].shape))


# In[ ]:


Xtrain_2.head()
print("Shape of Dataframe: {}".format(Xtrain_2.shape))


# In[ ]:


Xtrain_1.columns, Xtrain_2.columns, Xtest_1.columns, Xtest_2.columns


# ### ADDITIONAL FEATURE GENERATION

# In[ ]:


# Features constructed from previous case_count values

# Creating case_count lag features
def create_case_count_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] =                 gpby[target_col].shift(i).values + np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating case_count rolling mean features
def create_case_count_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmean',str(shift), str(w)])] =             gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).mean().values + np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating case_count exponentially weighted mean features
def create_case_count_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] =                 gpby[target_col].shift(s).ewm(alpha=a).mean().values + np.random.normal(scale=1.6, size=(len(df),))
    return df


# In[ ]:


# ONE HOT ENCODER OF CATEGORICAL FEATURES

def one_hot_encoder(df, ohe_cols=['dayofweek', 'quarter', 'month', 'dayofyear','dayofmonth', 'week']):
    '''
    One-Hot Encoder function
    '''
    print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
    df = pd.get_dummies(df, columns=ohe_cols)
    print('New df shape:{}'.format(df.shape))
    return df


# In[ ]:


Xtrain_1.shape, Xtrain_2.shape


# In[ ]:


Xtrain_1['segment'] = 1
Xtrain_2['segment'] = 2

Xtest_1['segment'] = 1
Xtest_2['segment'] = 2


# In[ ]:


train = pd.concat([Xtrain_1, Xtrain_2])

train['train_or_test'] = 'train'

train.shape


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


test = pd.concat([Xtest_1, Xtest_2])


test['train_or_test'] = 'test'

test.shape


# In[ ]:


# Taking log of 'case_count' column 

train['case_count'] = np.log1p(train.case_count.values)


# *WE WILL GENERATE NEW FEATURES AND THEN DO ONE HOT ENCODING *

# In[ ]:


df = pd.concat([train,test], sort=False)


# In[ ]:


df['application_date'] = df.index.get_level_values('application_date') 


# In[ ]:


df.application_date.min(), df.application_date.max()


# In[ ]:


# Time-based Validation set

# For validation to keep months also identical to test set we can choose period (same of 2018) as the validation set.

masked_series = (df['application_date'] >= '2018-07-06') & (df['application_date'] <= '2018-10-24')
masked_series2 = (df['application_date'] < '2018-07-06') & (df['application_date'] > '2018-10-24')
df.loc[(masked_series), 'train_or_test'] = 'val'
df.loc[(masked_series2), 'train_or_test'] = 'no_train'
print('Train shape: {}'.format(df.loc[df.train_or_test=='train',:].shape))
print('Validation shape: {}'.format(df.loc[df.train_or_test=='val',:].shape))
print('No train shape: {}'.format(df.loc[df.train_or_test=='no_train',:].shape))
print('Test shape: {}'.format(df.loc[df.train_or_test=='test',:].shape))


# In[ ]:


## Creating case_count lag, rolling mean, rolling median, ohe features of the above train set
train = create_case_count_lag_feats(train, gpby_cols=['segment'], target_col='case_count', 
                                    lags=[91,98,105,112,119,126,182,364,546,728])

train = create_case_count_rmean_feats(train, gpby_cols=['segment'], 
                                 target_col='case_count', windows=[364,546], 
                                 min_periods=10, win_type='triang')

train = create_case_count_ewm_feats(train, gpby_cols=['segment'], 
                               target_col='case_count', 
                               alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
                               shift=[91,98,105,112,119,126,182,364,546,728])


# In[ ]:


# Converting case_count of validation period to nan so as to resemble test period
train = df.loc[df.train_or_test.isin(['train','val']), :]
Y_val = train.loc[train.train_or_test=='val', 'case_count'].values.reshape((-1))
Y_train = train.loc[train.train_or_test=='train', 'case_count'].values.reshape((-1))
train.loc[train.train_or_test=='val', 'case_count'] = np.nan


# In[ ]:


# # Creating case_count lag, rolling mean, rolling median, ohe features of the above train set
train = create_case_count_lag_feats(train, gpby_cols=['segment'], target_col='case_count', 
                                    lags=[91,98,105,112,119,126,182,364,546,728])

train = create_case_count_rmean_feats(train, gpby_cols=['segment'], 
                                 target_col='case_count', windows=[364,546], 
                                 min_periods=10, win_type='triang')

train = create_case_count_ewm_feats(train, gpby_cols=['segment'], 
                               target_col='case_count', 
                               alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
                               shift=[91,98,105,112,119,126,182,364,546,728])

# One-Hot Encoding
train = one_hot_encoder(train) 

# Final train and val datasets
val = train.loc[train.train_or_test=='val', :]
train = train.loc[train.train_or_test=='train', :]
print('Train shape:{}, Val shape:{}'.format(train.shape, val.shape))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


avoid_cols = ['application_date', 'case_count', 'train_or_test', 'id', 'year','is_month_start']
cols = [col for col in train.columns if col not in avoid_cols]
print('No of training features: {} \nAnd they are:{}'.format(len(cols), cols))


# In[ ]:


train_x=train[cols]
test_x=val[cols]
train_y=Y_train
test_y=Y_val
train_x.shape, test_x.shape, train_y.shape, test_y.shape


# In[ ]:


train_x.fillna(0,inplace=True)
test_x.fillna(0,inplace=True)


# In[ ]:




