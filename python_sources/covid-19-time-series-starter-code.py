#!/usr/bin/env python
# coding: utf-8

# # Global Covid-19 Forecasting Using Time Series Prophet
# 
# ### We will use prophet time series to predict log(1+ConfirmedCases).
# 
# This is a simple starter code meant as an illustration for Prophet.

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


import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import pycountry
import plotly.express as px


# ## Import Data

# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
train.head()


# In[ ]:


train.info()


# In[ ]:



train.tail()


# In[ ]:


test.head()


# ### Remove Leaks
# We see that training data stops on March 27th and Test data starts on March 19th. This means, if we don't remove the overlapping dates then our model will likely overfit and will likely not generalize well in stage 2.

# In[ ]:


#train = train[train['Date'] < "2020-03-19"]
#train.sample(15)


# In[ ]:


from sklearn import preprocessing
#train['Lat'] = preprocessing.scale(train['Lat'])
#train['Long'] = preprocessing.scale(train['Long'])
#test['Lat'] = preprocessing.scale(test['Lat'])
#test['Long'] = preprocessing.scale(test['Long'])


# In[ ]:


# Format date
#train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))
#train["Date"]  = train["Date"].astype(int)
# drop nan's
#train = train.drop(['Province/State'],axis=1)
#train = train.dropna()
train.isnull().sum()


# In[ ]:


# Do same to Test data
#test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))
#test["Date"]  = test["Date"].astype(int)
# deal with nan's for lat and lon
#test["Lat"]  = test["Lat"].fillna(test['Lat'].mean())
#test["Long"]  = test["Long"].fillna(test['Long'].mean())
test.isnull().sum()


# In[ ]:


train['Country_Region'].unique()


# # Time Series Prophet Forecast for Germany
# 
# ## y = log(1+ConfirmedCases)
# 

# In[ ]:


# Time Series for ConfirmedCases
df = train[train['Country_Region'] == 'Germany']
#df = df[df['Date']]
#df1 = df.drop(['Id','Country/Region','Lat','Long'], axis=1)
confirmed=df.groupby('Date')['ConfirmedCases'].sum().to_frame().reset_index()

#confirmed = df1.drop(['Fatalities'], axis=1)
confirmed['ConfirmedCases'] = np.log(1+confirmed['ConfirmedCases'])
confirmed.plot()
#deaths = df.drop(['ConfirmedCases'], axis=1)


# In[ ]:



confirmed.columns = ['ds','y']
#confirmed['ds'] = confirmed['ds'].dt.date
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()


# In[ ]:


m = Prophet(interval_width=0.95)
m.fit(confirmed)
future = m.make_future_dataframe(periods=30)
future_confirmed = future.copy() # for non-baseline predictions later on
#future = future[future['ds'].unique()]
future


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[ ]:


confirmed_forecast_plot = m.plot(forecast)


# In[ ]:




