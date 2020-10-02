#!/usr/bin/env python
# coding: utf-8

# **Data: Accumulate Coronavirus Positive Cases (day-by-day) in Indonesia**
# 
# **Prediction Tool: FBProphet https://facebook.github.io/prophet/docs/quick_start.html**

# **1. Import all required libraries**

# In[ ]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from fbprophet import Prophet


# **2. Import Dataset**

# In[ ]:


data = pd.read_csv("../input/indonesia-coronavirus-cases/confirmed_acc.csv")

data.tail()


# **3. Plot dataset using Plotly**

# In[ ]:


end = datetime.datetime.now() - datetime.timedelta(1)
date_index = pd.date_range('2020-01-22', end)

fig = px.area(data, x=date_index, y='cases' )
fig.show()


# **4. Change DataFrame to FBProphet format**

# In[ ]:


df_prophet = data.rename(columns={"date": "ds", "cases": "y"})
df_prophet.tail()


# ### **5. Develop Prediction Model**
# 
# in this case we will use 
# * changepoint_prior_scale= 0.3
# * and changepoint_range = 0.95
# * only daily seasonality
# 
# for basic theory and detail configuration of FBProphet, read https://medium.com/future-vision/intro-to-prophet-9d5b1cbd674e
# 

# In[ ]:


from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot

m = Prophet(
    changepoint_prior_scale=0.3, # increasing it will make the trend more flexible
    changepoint_range=0.95, # place potential changepoints in the first 95% of the time series
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive'
)

m.fit(df_prophet)

future = m.make_future_dataframe(periods=100)
forecast = m.predict(future)


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)


# In[ ]:


fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[ ]:


forecast[100:170]

