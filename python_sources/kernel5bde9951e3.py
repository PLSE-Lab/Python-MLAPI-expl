#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


df_train= pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
df_test= pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[ ]:


100*(df_train.isnull().sum())/len(df_train)


# In[ ]:


100*(df_test.isnull().sum())/len(df_train)


# In[ ]:


df_train.rename(columns={'Country_Region':'Country'}, inplace=True)
df_test.rename(columns={'Country_Region':'Country'}, inplace=True)

df_train.rename(columns={'Province_State':'State'}, inplace=True)
df_test.rename(columns={'Province_State':'State'}, inplace=True)


# In[ ]:



df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)


# In[ ]:


y1_xTrain = df_train.iloc[:, -2]
y1_xTrain.head()


# In[ ]:


y2_xTrain = df_train.iloc[:, -1]
y2_xTrain.head()


# In[ ]:


EMPTY_VAL = "EMPTY_VAL"
def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state


# In[ ]:


c_train = df_train


# In[ ]:


c_train.head()


# In[ ]:


c_train['State'].fillna(EMPTY_VAL, inplace=True)
c_train['State'] = c_train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)


# In[ ]:


c_train.loc[:, 'Date'] = c_train.Date.dt.strftime("%m%d")
c_train["Date"]  = c_train["Date"].astype(int)


# In[ ]:


c_train.head()


# In[ ]:


c_test = df_test

c_test['State'].fillna(EMPTY_VAL, inplace=True)
c_test['State'] = c_test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

c_test.loc[:, 'Date'] = c_test.Date.dt.strftime("%m%d")
c_test["Date"]  = c_test["Date"].astype(int)


# In[ ]:


c_test.head()


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c_train.Country = le.fit_transform(c_train.Country)
c_train['State'] = le.fit_transform(c_train['State'])

c_train.head()


# In[ ]:


c_test.Country = le.fit_transform(c_test.Country)
c_test['State'] = le.fit_transform(c_test['State'])

c_test.head()


# In[ ]:


df_train.head()
df_train.loc[df_train.Country == 'Afghanistan', :]
df_test.tail()


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

from xgboost import XGBRegressor

countries = c_train.Country.unique()


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# In[ ]:


xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})


# In[ ]:


for country in countries:
    states = c_train.loc[c_train.Country == country, :].State.unique()
    #print(country, states)
    # check whether string is nan or not
    for state in states:
        X_xTrain_CS = c_train.loc[(c_train.Country == country) & (c_train.State == state), ['State','State_code', 'Country','Country_code', 'Date', 'ConfirmedCases', 'Fatalities']]
        
        y1_xTrain_CS = X_xTrain_CS.loc[:, 'ConfirmedCases']
        y2_xTrain_CS = X_xTrain_CS.loc[:, 'Fatalities']
        
        X_xTrain_CS = X_xTrain_CS.loc[:, ['State_code','Country','Country_code','State', 'Date']]
        
        X_xTrain_CS.Country = le.fit_transform(X_xTrain_CS.Country)
        X_xTrain_CS['State'] = le.fit_transform(X_xTrain_CS['State'])
        
        X_xTest_CS = c_test.loc[(c_test.Country == country) & (c_test.State == state), ['State','State_code', 'Country','Country_code', 'Date', 'ForecastId']]
        
        X_xTest_CS_Id = X_xTest_CS.loc[:, 'ForecastId']
        X_xTest_CS = X_xTest_CS.loc[:, ['State_code','Country','Country_code','State', 'Date']]
        
        X_xTest_CS.Country = le.fit_transform(X_xTest_CS.Country)
        X_xTest_CS['State'] = le.fit_transform(X_xTest_CS['State'])
        
        GBoost1 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
        GBoost1.fit(X_xTrain_CS, y1_xTrain_CS)
        y1_xpred = GBoost1.predict(X_xTest_CS)

        
        GBoost2 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
        GBoost2.fit(X_xTrain_CS, y2_xTrain_CS)
        y2_xpred = GBoost2.predict(X_xTest_CS)
        
        xdata = pd.DataFrame({'ForecastId': X_xTest_CS_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})
        xout = pd.concat([xout, xdata], axis=0)


# In[ ]:


xout.ForecastId = xout.ForecastId.astype('int')


# In[ ]:


xout.round().head()


# In[ ]:


xout.shape


# In[ ]:


xout.to_csv('submission.csv', index=False)


# In[ ]:




