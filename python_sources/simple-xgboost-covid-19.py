#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")


# In[ ]:


X_train = train.drop(["Fatalities", "ConfirmedCases"], axis=1)


# In[ ]:


X_train = X_train.drop(["Id"], axis=1)


# In[ ]:


X_train['Date']= pd.to_datetime(X_train['Date']) 


# In[ ]:


X_train = X_train.set_index(['Date'])


# In[ ]:


def create_time_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X


# In[ ]:


X_train_time = create_time_features(X_train)


# In[ ]:


X_train.drop("date", axis=1, inplace=True)


# In[ ]:


X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)
X_train.drop(['Province/State'],axis=1, inplace=True)


# In[ ]:


X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)
X_train.drop(['Country/Region'],axis=1, inplace=True)


# In[ ]:


y_train = train["Fatalities"]


# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)


# In[ ]:


reg.fit(X_train, y_train, verbose=True)


# In[ ]:


X_test = test.drop(["ForecastId"], axis=1)
X_test['Date']= pd.to_datetime(X_test['Date']) 
X_test = X_test.set_index(['Date'])
X_test_time = create_time_features(X_test)
X_test.drop("date", axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)
X_test.drop(['Province/State'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)
X_test.drop(['Country/Region'],axis=1, inplace=True)


# In[ ]:


X_test['Fatalities'] = reg.predict(X_test)


# In[ ]:


X_test

