#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight') # For plots


# In[ ]:


import pandas as pd
train = pd.read_csv("../input/application-train/train.csv")


# In[ ]:





# In[ ]:


train['application_date']=train['application_date'].astype('datetime64')


# In[ ]:


train1=train[train['segment']==1].groupby(['application_date'])['case_count'].sum().reset_index()


# In[ ]:


train2=train[train['segment']==2].groupby(['application_date'])['case_count'].sum().reset_index()


# In[ ]:


import seaborn as sns
sns.boxplot('case_count',data=train1)
Q1 = train1['case_count'].quantile(0.25)
Q3 = train1['case_count'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


train1 = train1[~((train1['case_count'] < (Q1 - 1.5 * IQR)) |(train1['case_count'] > (Q3 + 1.5 * IQR)))]


# 

# In[ ]:


train1


# # Data
# The data we will be using is hourly power consumption data from PJM. Energy consumtion has some unique charachteristics. It will be interesting to see how prophet picks them up.
# 
# Pulling the `PJM East` which has data from 2002-2018 for the entire east region.

# In[ ]:


def create_features(df, label=None):
    """
    Creates time series features from application_datetime index.
    """
    df = df.copy()
  
    df['dayofweek'] = df['application_date'].dt.dayofweek
    df['quarter'] = df['application_date'].dt.quarter
    df['month'] = df['application_date'].dt.month
    df['year'] = df['application_date'].dt.year
    df['dayofyear'] = df['application_date'].dt.dayofyear
    df['dayofmonth'] = df['application_date'].dt.day
    df['weekofyear'] = df['application_date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(train1, label='case_count')

features_and_target = pd.concat([X, y], axis=1)


# In[ ]:


X, y = create_features(train2, label='case_count')
features_and_target1 = pd.concat([X, y], axis=1)


# In[ ]:


features_and_target['ds']=train1['application_date'].astype('datetime64')


# In[ ]:


features_and_target1['ds']=train2['application_date'].astype('datetime64')


# In[ ]:


# See our features and target

features_and_target=features_and_target.rename({'case_count':'y'},axis=1)
features_and_target1=features_and_target1.rename({'case_count':'y'},axis=1)


# ## Plotting the Features to see trends
# - Power demand has strong daily and seasonal properties.
# - Day of week also seems to show differences in peaks

# # Train/Test Split
# Cut off the data after 2015 to use as our validation set. We will train on earlier data.

# In[ ]:


features_and_target


# In[ ]:



pjme_train = features_and_target[:600].copy()
pjme_test = features_and_target[600:].copy()
pjme_train1 = features_and_target1[:600].copy()
pjme_test1 = features_and_target1[600:].copy()


# # Simple Prophet Model
# - Prophet model expects the dataset to be named a specific way. We will rename our dataframe columns before feeding it into the model.

# In[ ]:


features_and_target.columns


# In[ ]:


exogenous_features=['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth',
       'weekofyear']


# In[ ]:


model=Prophet()
for feature in exogenous_features:
     model.add_regressor(feature)


# In[ ]:



model.fit(pjme_train[["ds", "y"] + exogenous_features])


# In[ ]:


model1 = Prophet()
for feature in exogenous_features:
     model1.add_regressor(feature)
model1.fit(pjme_train1[["ds", "y"] + exogenous_features])


# In[ ]:


# Predict on training set with model
pjme_test_fcst = model.predict(pjme_test)


# In[ ]:


pjme_test_fcst1=model1.predict(pjme_test1)


# In[ ]:


pjme_test_fcst.head()


# In[ ]:


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(pjme_test_fcst,
                 ax=ax)
plt.show()


# In[ ]:


# Plot the components of the model
fig = model.plot_components(pjme_test_fcst)


# # Compare Forecast to Actuals

# # Look at first month of predictions

# # Error Metrics
# 
# Our RMSE error is 43761675  
# Our MAE error is 5181.78  
# Our MAPE error is 16.5%
# 
# by comparison in the XGBoost model our errors were significantly less (8.9% MAPE):
# [Check that out here](https://www.kaggle.com/robikscube/hourly-time-series-forecasting-with-xgboost/)

# In[ ]:


mean_squared_error(y_true=pjme_test['y'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:


mean_absolute_error(y_true=pjme_test['y'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=pjme_test['y'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:


mean_absolute_percentage_error(y_true=pjme_test1['y'],
                   y_pred=pjme_test_fcst1['yhat'])


# # Adding Holidays
# Next we will see if adding holiday indicators will help the accuracy of the model. Prophet comes with a *Holiday Effects* parameter that can be provided to the model prior to training. 
# 
# We will use the built in pandas `USFederalHolidayCalendar` to pull the list of holidays

# In[ ]:


get_ipython().system('pip install holidays')


# In[ ]:


import holidays
holidays=pd.DataFrame({'ds':list(holidays.IND(years=[2017,2018,2019,2020]).keys()),'holiday':list(holidays.IND(years=[2017,2018,2019,2020]).values())})


# In[ ]:



model_with_holidays = Prophet(holidays=holidays)
for feature in exogenous_features:
    model_with_holidays.add_regressor(feature)
model_with_holidays.fit(pjme_train[["ds", "y"] + exogenous_features])
# Setup and train model with holidaysmodel = Prophet()


# ## Predict With Holidays

# In[ ]:


# Predict on training set with model
pjme_test_fcst_with_hols =     model_with_holidays.predict(pjme_test)


# ## Plot Holiday Effect

# In[ ]:


fig2 = model_with_holidays.plot_components(pjme_test_fcst_with_hols)


# # Error Metrics with Holidays Added
# Suprisingly the error has gotten worse after adding holidays.

# In[ ]:


mean_squared_error(y_true=pjme_test['y'],
                   y_pred=pjme_test_fcst_with_hols['yhat'])


# In[ ]:


mean_absolute_error(y_true=pjme_test['y'],
                   y_pred=pjme_test_fcst_with_hols['yhat'])


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=pjme_test['y'],
                   y_pred=pjme_test_fcst_with_hols['yhat'])


# In[ ]:


pjme_test_fcst_with_hols[pjme_test_fcst_with_hols['yhat']<0]


# In[ ]:


test = pd.read_csv("../input/test-data/test.csv")
test1=test[test['segment']==1].copy()
test2=test[test['segment']==2].copy()
test1['application_date']=test1['application_date'].astype('datetime64')
test2['application_date']=test2['application_date'].astype('datetime64')


# In[ ]:


def create_features(df, label=None):
    """
    Creates time series features from application_datetime index.
    """
    df = df.copy()
  
    df['dayofweek'] = df['application_date'].dt.dayofweek
    df['quarter'] = df['application_date'].dt.quarter
    df['month'] = df['application_date'].dt.month
    df['year'] = df['application_date'].dt.year
    df['dayofyear'] = df['application_date'].dt.dayofyear
    df['dayofmonth'] = df['application_date'].dt.day
    df['weekofyear'] = df['application_date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X= create_features(test1)


# In[ ]:


test1


# In[ ]:


X1= create_features(test2)


# In[ ]:


X1


# In[ ]:


test2


# In[ ]:


X['ds']=test1['application_date'].astype('datetime64')
X1['ds']=test2['application_date'].astype('datetime64')


# In[ ]:





# In[ ]:


predict=model.predict(X)


# In[ ]:


predict1=model.predict(X1)


# In[ ]:


predict[predict['yhat']<0]


# In[ ]:


predict1[predict1['yhat']<0]


# In[ ]:


final=pd.concat([predict,predict1])


# In[ ]:


final


# In[ ]:


test


# In[ ]:


final=final.reset_index()


# In[ ]:


test['case_count']=final['yhat']


# In[ ]:


test.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




