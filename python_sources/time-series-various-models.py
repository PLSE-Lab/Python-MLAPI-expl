#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')
meal = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/meal_info.csv')
center = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/fulfilment_center_info.csv')


# In[ ]:


train.head()


# In[ ]:


data = train.merge(meal, on='meal_id')


# In[ ]:


data = data.merge(center, on='center_id')


# In[ ]:


data.head()


# In[ ]:


data.nunique()


# In[ ]:


corr = data.corr()
import seaborn as sns
sns.heatmap(corr)


# In[ ]:


center_id = 55
meal_id = 1993


# In[ ]:


train_df = data[data['center_id']==center_id]
train_df = train_df[train_df['meal_id']==meal_id]


# In[ ]:


period = len(train_df)


# In[ ]:


train_df['Date'] = pd.date_range('2015-01-01', periods=period, freq='W')


# In[ ]:


train_df['Day'] = train_df['Date'].dt.day
train_df['Month'] = train_df['Date'].dt.month
train_df['Year'] = train_df['Date'].dt.year
train_df['Quarter'] = train_df['Date'].dt.quarter


# In[ ]:


train_df.head()


# # XGB Boost

# In[ ]:


xb_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine','center_type'])

xb_data = xb_data.set_index(['Date'])


# In[ ]:


x_train = xb_data.drop(columns='num_orders')
y_train = xb_data['num_orders']
y_train = np.log1p(y_train)
split_size = period-15
X_train = x_train.iloc[:split_size,:]
X_test = x_train.iloc[split_size:,:]
Y_train =  y_train.iloc[:split_size]
Y_test = y_train.iloc[split_size:]


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.plot(Y_train, label='training_data')
plt.plot(Y_test, label='Validation_test')
plt.legend(loc='best')


# In[ ]:


from xgboost import XGBRegressor
model_2 = XGBRegressor(
 learning_rate = 0.01,
 eval_metric ='rmse',
    n_estimators = 50000,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 1,
    gamma = 0.5
  
  
 )
#model.fit(X_train, y_train)
model_2.fit(X_train, Y_train, eval_metric='rmse', 
          eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)


# In[ ]:


a = (model_2.get_booster().best_iteration)
a


# In[ ]:


xgb_model = XGBRegressor(
     
     learning_rate = 0.01,
   
    n_estimators = a,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 1,
    gamma = 0.5)


# In[ ]:


xgb_model.fit(X_train, Y_train)


# In[ ]:


xgb_preds = xgb_model.predict(X_test)


# In[ ]:


xgb_preds = np.exp(xgb_preds)


# In[ ]:


train_df.tail()


# In[ ]:


xgb_preds = pd.DataFrame(xgb_preds)
xgb_preds.index = Y_test.index


# In[ ]:


Y_train = np.exp(Y_train)
Y_test = np.exp(Y_test)


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(Y_train, label='training_data')
plt.plot(Y_test, label='Validation_test')
plt.plot(xgb_preds, color='cyan', label='xgb_preds')
plt.legend(loc='best')


# # Light GBM Model

# In[ ]:


from lightgbm import LGBMRegressor
lgb_fit_params={"early_stopping_rounds":500, 
            "eval_metric" : 'rmse', 
            "eval_set" : [(X_test,Y_test)],
            'eval_names': ['valid'],
            'verbose':100
           }

lgb_params = {'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse',
 'verbose': 0,
 'bagging_fraction': 0.8,
 'bagging_freq': 1,
 'lambda_l1': 0.01,
 'lambda_l2': 0.01,
 'learning_rate': 0.001,
 'max_bin': 255,
 'max_depth': 6,
 'min_data_in_bin': 1,
 'min_data_in_leaf': 1,
 'num_leaves': 31}

Y_train = np.log1p(Y_train)
Y_test = np.log1p(Y_test)


# In[ ]:


clf_lgb = LGBMRegressor(n_estimators=10000, **lgb_params, random_state=123456789, n_jobs=-1)
clf_lgb.fit(X_train, Y_train, **lgb_fit_params)


# In[ ]:


lgb_model = LGBMRegressor(bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.01,
              lambda_l2=0.01, learning_rate=0.01, max_bin=255, max_depth=6,
              metric='rmse', min_data_in_bin=1, min_data_in_leaf=1,
              n_estimators=10000, objective='regression',
              random_state=123456789, verbose=0)


# In[ ]:


lgb_model.fit(X_train,Y_train)


# In[ ]:


lgm_preds = lgb_model.predict(X_test)
lgm_preds = np.exp(lgm_preds)


# In[ ]:


lgm_preds = pd.DataFrame(lgm_preds)
lgm_preds.index = Y_test.index


# In[ ]:


Y_train = np.exp(Y_train)
Y_test = np.exp(Y_test)


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.legend(loc='best')


# # Cat_Regressor

# In[ ]:


from catboost import CatBoostRegressor
Y_train = np.log1p(Y_train)
Y_test = np.log1p(Y_test)

cat_model=CatBoostRegressor()
cat_model.fit(X_train, Y_train)


# In[ ]:


cat_preds = cat_model.predict(X_test)
cat_preds = np.exp(cat_preds)


# In[ ]:


cat_preds = pd.DataFrame(cat_preds)
cat_preds.index = Y_test.index
Y_train = np.exp(Y_train)
Y_test = np.exp(Y_test)


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.plot(cat_preds, color='green', label='cat_prediction')
plt.legend(loc='best')


# # Prophet model

# In[ ]:


prophet_data = train_df[['Date','num_orders']]
prophet_data.index = xb_data.index
prophet_data = prophet_data.iloc[:split_size,:]


# In[ ]:


prophet_data =prophet_data.rename(columns={'Date':'ds',
                             'num_orders':'y'})
prophet_data.head()


# In[ ]:


from fbprophet import Prophet
m = Prophet(growth='linear',
            seasonality_mode='multiplicative',
#            changepoint_prior_scale = 30,
           seasonality_prior_scale = 35,
           holidays_prior_scale = 10,
           daily_seasonality = True,
           weekly_seasonality = False,
           yearly_seasonality= False,
           ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=30
            
            ).add_seasonality(
                name='weekly',
                period=7,
                fourier_order=55
            ).add_seasonality(
                name='yearly',
                period=365.25,
                fourier_order=20
            )
        
m.fit(prophet_data)


# In[ ]:


future = m.make_future_dataframe(periods=15, freq='W')


# In[ ]:


forecast = m.predict(future)
# forecast['yhat'] = np.exp(forecast['yhat'])
# forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
# forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig2 = m.plot_components(forecast)


# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# In[ ]:


prophet_preds = forecast['yhat'].iloc[split_size:]
prophet_preds.index = Y_test.index


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.plot(prophet_preds, color='green', label='prophet_prediction')
plt.plot(cat_preds, color='blue', label='cat_prediction')
plt.legend(loc='best')


# In[ ]:


plt.figure(figsize=(20,5))
# plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.plot(prophet_preds, color='green', label='prophet_prediction')
plt.plot(cat_preds, color='blue', label='cat_prediction')
plt.legend(loc='best')


# # Combine Forecast

# In[ ]:


a = np.array(prophet_preds)
b = np.array(lgm_preds)
c = np.array(xgb_preds)
d = np.array(cat_preds)
final_preds =  (b*0.8)+ (d*0.2) 
final_preds = (final_preds*0.4) + (a*0.6)


# In[ ]:


final_preds[6]


# In[ ]:


final_preds = pd.DataFrame(final_preds[6])
final_preds.index = Y_test.index


# In[ ]:


final_preds = pd.DataFrame(final_preds)
final_preds.index = Y_test.index
plt.figure(figsize=(20,5))
# plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='orange', label='light_lgm_prediction')
plt.plot(prophet_preds, color='green', label='prophet_prediction')
plt.plot(final_preds, color='red',linestyle='--', label='final_prediction')
plt.plot(cat_preds, color='blue', label='cat_prediction')
plt.legend(loc='best')


# In[ ]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, final_preds, squared=False))

