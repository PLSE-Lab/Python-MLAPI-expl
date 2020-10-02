#!/usr/bin/env python
# coding: utf-8

# Loading data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test= pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
train.head()


# In[ ]:


# converting the dtypes to datetime format
train["Date"] = pd.to_datetime(train["Date"])
test['Date'] = pd.to_datetime(test['Date'])

train = train.set_index(train['Date'])
test = test.set_index(test['Date'])

france= train[(train.Country_Region== 'France')]
france.tail()


# In[ ]:


france.info()


# In[ ]:


france.shape


# In[ ]:


france_1= france[['Date', 'ConfirmedCases']]
france_1['ds']= france_1['Date']
france_1['y']= france_1['ConfirmedCases']
france_1.drop(columns=['Date','ConfirmedCases'], inplace= True)
france_1.head()


# # Visualisation

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
ax= france_1.set_index('ds').plot(figsize= (15,8))
ax.set_xlabel('Date')
ax.set_ylabel('Fatalities')

plt.show()


# # Prediction

# In[ ]:


from fbprophet import Prophet
from fbprophet.plot import plot_plotly


# In[ ]:


model_f=Prophet(interval_width = 0.95)
model_f.fit(france_1)


# In[ ]:


future= model_f.make_future_dataframe(periods= 100)
future.tail()


# In[ ]:


forecast= model_f.predict(future)
forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig1= model_f.plot(forecast, uncertainty= True)


# In[ ]:


model_f.plot_components(forecast)


# In[ ]:


from fbprophet.plot import add_changepoints_to_plot
fig = model_f.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model_f, forecast)


# In[ ]:


model_f.changepoints


# In[ ]:


pro_change= Prophet(changepoint_range=0.9)
forecast = pro_change.fit(france_1).predict(future)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)


# In[ ]:


pro_change= Prophet(n_changepoints=20, yearly_seasonality=True)
forecast = pro_change.fit(france_1).predict(future)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

