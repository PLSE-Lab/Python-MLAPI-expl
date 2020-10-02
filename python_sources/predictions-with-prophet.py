#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from fbprophet import Prophet
import logging
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/industrial-production-index-in-usa/INDPRO.csv')
df.head()


# In[ ]:


df.columns = ['Date', 'IPI']
df.head()


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index ('Date', inplace = True)
df.index


# In[ ]:


df_new = df['1998-01-01':]
df_new.head()


# In[ ]:


# to check NAs
df_new.info()
df_new.isnull().sum()


# In[ ]:


df_new.describe().transpose()


# In[ ]:


f, ax = plt.subplots(figsize = (16,10))
ax.plot(df_new, c = 'r');


# In[ ]:


df_new.columns


# In[ ]:


df_new.index


# Prophet requires the date column to be named ds and the feature column to be named y:

# In[ ]:


df_new = df_new.reset_index()
df_new.columns


# In[ ]:


df_new.columns = ['ds', 'y'] 
df_new.head()


# Now, we need to define a training set. In order to do that, we will leave out last 30 observations to predict and validate our forecast. And then, we fit the model to the data, and make the forecast!

# In[ ]:


#periods = 30
#train_df = df_new[:-periods]


# In[ ]:


m = Prophet()
m.fit(df_new)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)

