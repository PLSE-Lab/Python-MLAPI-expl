#!/usr/bin/env python
# coding: utf-8

# This Kernel is the succeeding file for the following great kernel:
# 
# >- [ASHRAE: EDA File](https://www.kaggle.com/trehansalil1/ashrae-eda)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
import lightgbm as lgb
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix
precision_recall_fscore_support(",", "accuracy_score,", "cohen_kappa_score,", "f1_score,", "classification_report")
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv',parse_dates=['timestamp'])


# In[ ]:


weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv',parse_dates=['timestamp'])


# In[ ]:


weather_train.head()


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


# In[ ]:


weather_test = reduce_mem_usage(weather_test)
weather_train = reduce_mem_usage(weather_train)


# In[ ]:


weather_test.head()


# In[ ]:


weather_train.head()


# In[ ]:


weather_train = weather_train.groupby('site_id').apply(lambda x: x.set_index('timestamp').interpolate(method='time',limit_direction='both').reset_index()).drop(columns='site_id').reset_index().drop(columns='level_1')


weather_train.isnull().sum()


# In[ ]:


weather_test = weather_test.groupby('site_id').apply(lambda x: x.set_index('timestamp').interpolate(method='time',limit_direction='both').reset_index()).drop(columns='site_id').reset_index().drop(columns='level_1')


# In[ ]:


weather_test.isnull().sum()


# In[ ]:


weather_train.head()


# In[ ]:


weather_train['cloud_coverage'] = weather_train['cloud_coverage'].round()
weather_train['wind_direction'] = weather_train['wind_direction'].round()
weather_train['precip_depth_1_hr'] = weather_train['precip_depth_1_hr'].round()

weather_test['cloud_coverage'] = weather_test['cloud_coverage'].round()
weather_test['wind_direction'] = weather_test['wind_direction'].round()
weather_test['precip_depth_1_hr'] = weather_test['precip_depth_1_hr'].round()


# In[ ]:


weather_test.groupby('site_id').mean()


# In[ ]:


weather_test[weather_test['sea_level_pressure'].isnull()]['site_id'].unique()


# In[ ]:


weather_test[weather_test['wind_direction'].isnull()]['site_id'].unique()


# In[ ]:


weather_train['aday'] = weather_train['timestamp'].dt.day
weather_train['amonth'] = weather_train['timestamp'].dt.month
weather_train['ayear'] = weather_train['timestamp'].dt.year
weather_train['ahour'] = weather_train['timestamp'].dt.hour


# In[ ]:


weather_test['aday'] = weather_test['timestamp'].dt.day
weather_test['amonth'] = weather_test['timestamp'].dt.month
weather_test['ayear'] = weather_test['timestamp'].dt.year
weather_test['ahour'] = weather_test['timestamp'].dt.hour


# In[ ]:


def sea_impute(weather_test):
    
    
    df_test = weather_test.loc[weather_test.site_id==5,:].drop(columns='timestamp').reset_index().drop(columns='index')
    df_train = weather_test.loc[weather_test.site_id!=5,:].drop(columns='timestamp').reset_index().drop(columns='index')
    X = df_train.drop(columns='sea_level_pressure')
    y = df_train['sea_level_pressure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = lgb.LGBMRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print('The mean squared error for sea_level_pressure is: ',mean_squared_error(y_test, y_pred))

    model.fit(X ,y)
    y_pred = model.predict(df_test.drop(columns='sea_level_pressure'))
    df_test['sea_level_pressure'] = pd.Series(y_pred)
    df = pd.merge(df_train, df_test, how='outer')

    del df_test, df_train, X_train, X_test, y_train, y_test, y_pred, X, y

    weather_test = df
    
    return(weather_test)

def cloud_impute(weather_test):
    from sklearn.metrics import classification_report, cohen_kappa_score
    df_test = weather_test.loc[(weather_test.site_id == 7) | (weather_test.site_id == 11), :].reset_index().drop(columns='index')
    df_train = weather_test.loc[(weather_test.site_id != 7) & (weather_test.site_id != 11), :].reset_index().drop(columns='index')
    df_test.dropna(subset=['precip_depth_1_hr'], inplace=True)
    df_train.dropna(subset=['precip_depth_1_hr'], inplace=True)
    X = df_train.drop(columns='cloud_coverage').dropna()
    y = df_train['cloud_coverage']
    smote = SMOTE('minority')
    X, y = smote.fit_sample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = lgb.LGBMClassifier(objective='multiclass', reg_alpha= 1, reg_lambda=5, learning_rate=1.5, n_extimator=50)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('The cohen_kappa_score & f1_score for cloud_coverage is: ',cohen_kappa_score(y_test, y_pred)
          , f1_score(y_test, y_pred, average='micro'))
    print('Report for cloud_coverage: ')
    print(classification_report(y_test, y_pred))
    model.fit(X, y)
    y_pred = model.predict(df_test.drop(columns='cloud_coverage'))

    df_test['cloud_coverage'] = pd.Series(y_pred)
    df = pd.merge(df_train,df_test, how='outer')
    k = weather_test.loc[(weather_test.site_id == 1) | (weather_test.site_id == 12) | (weather_test.site_id == 5)
                         , :].reset_index().drop(columns='index')

    df = pd.merge(df,k, how='outer')

    del df_test, df_train, X_train, X_test, y_train, y_test, y_pred, X, y, k

    weather_test = df
    
    return(weather_test)

def precip_impute(weather_test):
    df_test = weather_test.loc[(weather_test.site_id == 1) | (weather_test.site_id == 12) | (weather_test.site_id == 5), :].reset_index().drop(columns='index')
    df_train = weather_test.loc[(weather_test.site_id != 1) & (weather_test.site_id != 12) & (weather_test.site_id != 5), :].reset_index().drop(columns='index')
    X = df_train.drop(columns='precip_depth_1_hr')
    y = df_train['precip_depth_1_hr']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    model = lgb.LGBMRegressor(learning_rate=0.08, n_estimators=10000, reg_alpha=1
                              , reg_lambda=2, lambda_l1 = 0.3)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('The mean squared error for precip_depth_1_hr is: ', mean_squared_error(y_test, y_pred))
    y_pred = np.array([round(i) for i in y_pred])
    print('The corrected mean squared error for precip_depth_1_hr is: ', mean_squared_error(y_test, y_pred))

    model.fit(X,y)
    y_pred = model.predict(df_test.drop(columns='precip_depth_1_hr'))
    y_pred = np.array([round(i) for i in y_pred])

    df_test['precip_depth_1_hr'] = pd.Series(y_pred)
    df = pd.merge(df_train, df_test, how='outer')
    #del df_test, df_train, X_train, X_test, y_train, y_test, y_pred, X, y

    weather_test = df
    
    return(weather_test)

#del df


# In[ ]:


weather_test = sea_impute(weather_test)


# In[ ]:


weather_test = cloud_impute(weather_test)


# In[ ]:


weather_test = precip_impute(weather_test)


# In[ ]:


weather_test.groupby('site_id').mean().sort_values(by='cloud_coverage')


# In[ ]:


weather_test.isnull().sum()[weather_test.isnull().sum()!=0]


# In[ ]:


weather_train = sea_impute(weather_train)


# In[ ]:


weather_train = cloud_impute(weather_train)


# In[ ]:


weather_train = precip_impute(weather_train)


# In[ ]:


weather_train.isnull().sum()[weather_train.isnull().sum()!=0]


# In[ ]:


weather_test.to_csv('weather_test_s.csv')


# In[ ]:


weather_train.to_csv('weather_train_s.csv')


# In[ ]:




