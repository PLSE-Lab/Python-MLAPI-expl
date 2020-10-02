#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Date & Time
from datetime import date, datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

# plt.style.use('ggplot')
plt.style.use('seaborn-white')
font = {
    'family' : 'normal',
    'weight' : 'bold',
    'size'   : 13
}
plt.rc('font', **font)


# In[ ]:


df = pd.read_csv('../input/michael-jordan-kobe-bryant-and-lebron-james-stats/allgames_stats.csv')
df.head()


# In[ ]:


from fbprophet import Prophet


# In[ ]:


df1=df.rename(columns={"Date": "ds", "FG": "y"})
df1


# In[ ]:


m = Prophet()
m.fit(df1)


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

