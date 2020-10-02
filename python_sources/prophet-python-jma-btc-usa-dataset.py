#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')


import plotly.tools as tls
import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

print(__version__) # requires version >= 1.9.0
cf.go_offline()


# In[ ]:


df = pd.read_csv('../input/btcusd-dataset/btc.csv')
df.head()


# In[ ]:


df1=df.rename(columns={"Date": "ds", "Volume": "y"})
df1


# In[ ]:


df2=df1.drop(["High", "Low","Mid","Last","Bid","Ask"], axis = 1)
df2


# In[ ]:


import pandas as pd
from fbprophet import Prophet
m = Prophet()
m.fit(df2)


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
