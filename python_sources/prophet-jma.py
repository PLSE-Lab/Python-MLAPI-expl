#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from fbprophet import Prophet


# In[ ]:


df = pd.read_csv('../input/example-wp-log-peyton-manningcsv/example_wp_log_peyton_manning.csv')
df.head()


# In[ ]:


m = Prophet()
m.fit(df)


# In[ ]:


future = m.make_future_dataframe(periods=365)
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig1 = m.plot(forecast)


# In[ ]:


fig2 = m.plot_components(forecast)


# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)

