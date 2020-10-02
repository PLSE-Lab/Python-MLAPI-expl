#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m

from datetime import datetime

import os
import glob


import folium 

import geopandas

from folium import plugins


# In[ ]:


pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# Code for displaying plotly express plot
def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


# In[ ]:


from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.express as px
configure_plotly_browser_state()
from IPython.display import IFrame


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


# Building and fitting Random Forest
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


covid_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
covid_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
data_path = '/kaggle/input/'
weather_data = pd.read_csv('../input/covid19formattedweatherjan22march24/covid_dataset.csv')
'''
pollution_data = pd.read_csv('../input/pollution-by-country-for-covid19-analysis/region_pollution.csv')

response_tracker_date = pd.read_excel(data_path + 'oxford-covid19-government-response-tracker/OxCGRT_Download_latest_data.xlsx')

cord_metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
'''


# In[ ]:


submission = pd.read_csv(data_path + '/covid19-global-forecasting-week-2/submission.csv')
submission.head()


# In[ ]:


covid_train.head(10)
covid_train.shape
train = covid_train
test = covid_test
#covid_train.info()


# In[ ]:


covid_train.rename(columns={'Province_State':'Province'}, inplace=True)
covid_train.rename(columns={'Country_Region':'Country'}, inplace=True)
covid_train.rename(columns={'Id':'ForecastId'}, inplace=True)

covid_test.rename(columns={'Province_State':'Province'}, inplace=True)
covid_test.rename(columns={'Country_Region':'Country'}, inplace=True)


# In[ ]:


covid_train['Date'] = pd.to_datetime(covid_train['Date'])
covid_test['Date'] = pd.to_datetime(covid_test['Date'])


# In[ ]:


covid_train = covid_train.set_index(['Date'])
covid_test = covid_test.set_index(['Date'])


# In[ ]:


print ('Training Data provided from', train['Date'].min(),'to ', train['Date'].max() )

print ('Test Data provided from', test['Date'].min(),'to ', test['Date'].max() )

print ('Weather Data provided from 2020-01-22 to 2020-03-24')


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


covid_train['Province'] = covid_train['Province'].fillna(covid_train['Country'])
covid_test['Province'] = covid_test['Province'].fillna(covid_test['Country'])


# In[ ]:


create_time_features(covid_train).head()
create_time_features(covid_test).head()


# In[ ]:


covid_train.isnull().values.any()
covid_train.isnull().sum()
covid_test.isnull().values.any()
covid_test.isnull().sum()


# In[ ]:


configure_plotly_browser_state()
fig = px.scatter(covid_train, x="date", y="ConfirmedCases",   
                 color="Country",
                 hover_name="Province")
fig.show()


# In[ ]:


configure_plotly_browser_state()
fig = px.scatter(covid_train, x="date", y="Fatalities",   
                 color="Country",
                 hover_name="Province")
fig.show()


# In[ ]:


configure_plotly_browser_state()
fig = px.scatter(covid_train.dropna(), y="ConfirmedCases", x="Fatalities",   
                 color="Province",
                 hover_name="Country", 
                log_x=True, size_max=60
                )
fig.show()


# In[ ]:


grpbydate = covid_train.groupby(['date','Country', 'Province'])['ConfirmedCases', 'Fatalities'].sum().reset_index().sort_values('date', ascending = True)


# In[ ]:


grpbydate[grpbydate['Country']=='US']


# In[ ]:


grpbydate['Country'].nunique()


# In[ ]:


weather_data.rename(columns={'Province/State':'Province'}, inplace=True)
weather_data.rename(columns={'Country/Region':'Country'}, inplace=True)
weather_data['Province'] = weather_data['Province'].fillna(weather_data['Country'])


# In[ ]:


weather_data[(weather_data['Country'] == 'US') ].head(100)


# In[ ]:


weather_data_df = geopandas.GeoDataFrame(
    weather_data, geometry=geopandas.points_from_xy(weather_data.long, weather_data.lat))


# In[ ]:


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
weather_data_df.plot(ax=world.plot(figsize=(28, 12)), marker='o', color='#fb5599', markersize=10);
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
covid_train['country_encoded'] = labelencoder.fit_transform(covid_train['Country'])
covid_test['country_encoded'] = labelencoder.fit_transform(covid_test['Country'])


# In[ ]:


covid_train.describe(include=('int64','float64'))


# In[ ]:


covid_test.describe(include = ('int64','float64'))


# In[ ]:


covid_train.head()
covid_test.head()


# ### Train Test Split

# In[ ]:


train_y_inf = covid_train["ConfirmedCases"]
train_y_ft = covid_train["Fatalities"]
train_x = covid_train
train_x.drop(["ConfirmedCases","Fatalities", "Country","date", "Province"], axis=1, inplace=True)


# In[ ]:


test_x = covid_test
test_x.drop(["Country","date", "Province"], axis=1, inplace=True)


# ### Time Series Prophet

# In[ ]:


from fbprophet import Prophet


# In[ ]:


train_ts= pd.DataFrame()
train_ts['ds'] = pd.to_datetime(train["Date"])
train_ts['y']=train["ConfirmedCases"]
indexedData = train_ts.set_index('ds')
indexedData.tail()


# In[ ]:


m = Prophet(yearly_seasonality= True,
    weekly_seasonality = True,
    daily_seasonality = True,
    seasonality_mode = 'multiplicative')
m.fit(train_ts)
future = m.make_future_dataframe(periods=14)
forecast = m.predict(future)


# In[ ]:


m.plot_components(forecast)


# In[ ]:


py.iplot([
    go.Scatter(x=train_ts['ds'], y=train_ts['y'], name='y'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')
])


# In[ ]:


from fbprophet.plot import plot_plotly

py.init_notebook_mode()

fig = plot_plotly(m, forecast)  
py.iplot(fig)


# In[ ]:


train_ts1= pd.DataFrame()
train_ts1['ds'] = pd.to_datetime(train["Date"])
train_ts1['y']=train["Fatalities"]
indexedData = train_ts1.set_index('ds')
indexedData.tail()


# In[ ]:


m = Prophet(yearly_seasonality= True,
    weekly_seasonality = True,
    daily_seasonality = True,
    seasonality_mode = 'multiplicative')
m.fit(train_ts1)
future1 = m.make_future_dataframe(periods=14)
forecast1 = m.predict(future1)


# In[ ]:


m.plot_components(forecast1)


# In[ ]:


py.iplot([
    go.Scatter(x=train_ts1['ds'], y=train_ts1['y'], name='y'),
    go.Scatter(x=forecast1['ds'], y=forecast1['yhat'], name='yhat'),
    go.Scatter(x=forecast1['ds'], y=forecast1['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast1['ds'], y=forecast1['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast1['ds'], y=forecast1['trend'], name='Trend')
])


# In[ ]:


#py.init_notebook_mode()
fig = plot_plotly(m, forecast1)  
py.iplot(fig)


# In[ ]:


ts_submission = pd.DataFrame()
ts_submission['ForecastId'] = forecast.index
ts_submission['ConfirmedCases'] = forecast['yhat']
ts_submission['Fatalities'] = forecast1['yhat']


# In[ ]:


ts_submission['ConfirmedCases'] = ts_submission['ConfirmedCases'].astype(int)
ts_submission['Fatalities'] = ts_submission['Fatalities'].astype(int)
ts_submission.tail()


# In[ ]:


#ts_submission.to_csv('submission.csv',index = False)


# ### XGBoost Regressor

# In[ ]:


import xgboost 
xg_reg = xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, 
                          alpha = 10, n_estimators = 100)


# In[ ]:


xg_reg.fit(train_x,train_y_inf)
preds = xg_reg.predict(test_x)


# In[ ]:


xg_reg.fit(train_x,train_y_ft)
preds1 = xg_reg.predict(test_x)


# In[ ]:


xgb_submission = pd.DataFrame()
xgb_submission['ForecastId'] = test_x['ForecastId']
xgb_submission['ConfirmedCases'] = preds
xgb_submission['Fatalities'] = preds1


# In[ ]:


xgb_submission['ConfirmedCases'] = xgb_submission['ConfirmedCases'].astype(int)
xgb_submission['Fatalities']  = xgb_submission['Fatalities'].astype(int)


# In[ ]:


xgb_submission.to_csv('submission.csv', index = False)


# In[ ]:



xgdmat = xgboost.DMatrix(train_x,train_y_inf) # Create our DMatrix to make XGBoost more efficient
xgdmat
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'reg:linear', 'max_depth':3, 'min_child_weight':1} 
# Grid Search CV optimized settings
get_ipython().run_line_magic('time', "cv_xgb = xgboost.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5, metrics = ['logloss'], early_stopping_rounds = 100) # Look for early stopping that minimizes error,")
# Make sure you enter metrics inside a list or you may encounter issues!
cv_xgb.head()


# In[ ]:


covid_xgb = xgboost.train(our_params, xgdmat, num_boost_round = 50)
#testdmat = xgboost.DMatrix(x_test_infected)


# In[ ]:


xgboost.plot_tree(covid_xgb,num_trees=0)
plt.rcParams['figure.figsize'] = [100, 50]
plt.show()


# In[ ]:


xgboost.plot_importance(covid_xgb)
plt.rcParams['figure.figsize'] = [17, 15]
plt.show()


# ### Random Forest Regressor

# In[ ]:


forest = RandomForestRegressor(max_depth = 10, n_estimators = 1000, random_state = 2020)
covid_rf = forest.fit(train_x, train_y_inf)
print(covid_rf.score(train_x, train_y_inf))


# ### Model Prediction

# In[ ]:


rf_test_pred_infected = covid_rf.predict(test_x)
#covid_rf_mse = mean_squared_error(test_y_inf, rf_test_pred_infected)
#print (covid_rf_mse)
rf_test_pred_infected.astype(int)
rf_test_pred_infected[rf_test_pred_infected<0]=0


# In[ ]:


forest1 = RandomForestRegressor(max_depth = 10, n_estimators = 1000, random_state = 2020)
covid_rf1 = forest1.fit(train_x, train_y_ft)
print(covid_rf1.score(train_x, train_y_ft))


# In[ ]:


rf_test_pred_fatality = covid_rf1.predict(test_x)

rf_test_pred_fatality.astype(int)
rf_test_pred_fatality[rf_test_pred_fatality<0]=0


# In[ ]:


submission = pd.DataFrame()
submission['ForecastId'] = test_x['ForecastId']
submission['ConfirmedCases'] = rf_test_pred_infected
submission['Fatalities'] = rf_test_pred_fatality


# In[ ]:


submission['ConfirmedCases'] = submission['ConfirmedCases'].astype(int)
submission['Fatalities'] = submission['Fatalities'].astype(int)


# #### Submission

# In[ ]:


#submission.to_csv('submission.csv', index = False)


# In[ ]:


#submission.head()


# # Datasets Acknowledgement
# 
# https://www.kaggle.com/hbfree/covid19formattedweatherjan22march24
# 
# 
