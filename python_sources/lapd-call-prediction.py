#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


# In[ ]:


df_by_hour = pd.read_csv("../input/total_calls_per_hour.csv")


# In[ ]:


df_by_hour["ds"] = pd.date_range(min(df_by_hour["ds"]), max(df_by_hour["ds"]), freq='H')


# In[ ]:


df_by_hour.head()


# In[ ]:


df_by_hour.tail()


# In[ ]:


m = Prophet()
m.fit(df_by_hour)


# In[ ]:


future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


fig1 = m.plot(forecast)


# In[ ]:


fig2 = m.plot_components(forecast)


# In[ ]:


py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# In[ ]:


df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_cv.head()


# In[ ]:


df_p = performance_metrics(df_cv)
df_p.head()


# In[ ]:


fig = plot_cross_validation_metric(df_cv, metric='rmse')

