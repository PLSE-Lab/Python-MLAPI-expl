#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
from fastai.tabular import add_datepart
import plotly.graph_objs as go


# In[ ]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')


# In[ ]:


train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))


# In[ ]:


test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))


# In[ ]:


test_date_min - train_date_max


# In[ ]:


def categoricalToInteger(df):
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])
    return df


# In[ ]:


add_datepart(df_train, 'Date', drop=False)
df_train.drop('Elapsed', axis=1, inplace=True) 


# In[ ]:


df_train = categoricalToInteger(df_train)
#matrix = df_train[['day','Country_Region','Province_State','ConfirmedCases','Fatalities']]


# In[ ]:


def lag_feature(df, lags, col):
    tmp = df[['Dayofyear','Country_Region','Province_State',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['Dayofyear','Country_Region','Province_State', col+'_lag_'+str(i)]
        shifted['Dayofyear'] += i
        df = pd.merge(df, shifted, on=['Dayofyear','Country_Region','Province_State'], how='left')
    return df


# In[ ]:


df_train = lag_feature(df_train,[1,2,3,6,11],'ConfirmedCases')


# In[ ]:


df_train = lag_feature(df_train,[1,2,3,6,11],'Fatalities')


# In[ ]:


df_train.columns


# In[ ]:


lags = [1,2,3,6,11]
features = [
        'ConfirmedCases',
        'Fatalities',
        'Year',
        'Month',
        'Week',
        'Day',
        'Dayofweek',
        'Dayofyear',
        'Is_month_end',
        'Is_month_start',
        'Is_quarter_end',
        'Is_quarter_start',
        'Is_year_end',
        'Is_year_start'
]
for lag in lags:
    features.append("ConfirmedCases_lag_"+str(lag))
    features.append("Fatalities_lag_"+str(lag))
    #features.append("day_avg_cases_lag_"+str(lag))
        
corr_matrix = df_train[features].corr()
corr_matrix["ConfirmedCases"].sort_values(ascending=False)


# In[ ]:


corr_matrix["Fatalities"].sort_values(ascending=False)


# In[ ]:


df_train = df_train[df_train['Date']<test_date_min]


# In[ ]:


df_train.drop(['Id','Date'], axis=1,inplace=True)


# In[ ]:


df_train = df_train[[c for c in df_train if c not in ['ConfirmedCases', 'Fatalities']] + ['ConfirmedCases', 'Fatalities']]


# In[ ]:


train = df_train.values


# In[ ]:


X_train, y_train = train[:,:-2], train[:,-2:]


# In[ ]:


model1 = XGBRegressor(n_estimators=1000)
model1.fit(X_train, y_train[:,0])


# In[ ]:


model2 = XGBRegressor(n_estimators=1000)
model2.fit(X_train, y_train[:,1])


# In[ ]:


from xgboost import plot_importance
import matplotlib.pyplot as plt

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


# In[ ]:


plot_features(model1, (10,14))


# In[ ]:


plot_features(model2, (10,14))


# In[ ]:


add_datepart(df_test, 'Date', drop=False)
df_test.drop('Elapsed', axis=1, inplace=True) 


# In[ ]:


df_test = categoricalToInteger(df_test)


# In[ ]:


df_test.drop(['ForecastId','Date'], axis=1,inplace=True)


# In[ ]:


cols = list(set(df_train.columns[:-2])-set(df_test.columns))
for col in cols:
    df_test[col] = 0 


# In[ ]:


df_test = df_test[['Province_State', 'Country_Region', 'Year', 'Month', 'Week', 'Day',
       'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
       'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
       'ConfirmedCases_lag_1', 'ConfirmedCases_lag_2', 'ConfirmedCases_lag_3',
       'ConfirmedCases_lag_6', 'ConfirmedCases_lag_11', 'Fatalities_lag_1',
       'Fatalities_lag_2', 'Fatalities_lag_3', 'Fatalities_lag_6',
       'Fatalities_lag_11']]


# In[ ]:


y_pred1 = model1.predict(df_test.values)
y_pred2 = model2.predict(df_test.values)


# In[ ]:


df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv') 
df_submit.ConfirmedCases = y_pred1
df_submit.Fatalities = y_pred2
df_submit.to_csv(r'submission.csv', index=False)


# In[ ]:




