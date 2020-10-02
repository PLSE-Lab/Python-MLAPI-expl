#!/usr/bin/env python
# coding: utf-8

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


# ## Importing Plotly to Create Time Series Charts

# In[ ]:


import plotly.graph_objects as go


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# ## Loading the DataFrame

# In[ ]:


rates = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')


# In[ ]:


rates.head()


# ## Time Series Charts

# #### Let's take a look at how the exchange rate fluctuated for the following currencies: **CAD/USD**, **CNY/USD**, **EUR/USD** and **GBP/USD**.

# ### Daily Exchange Rates (2000 - 2019)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['CANADA - CANADIAN DOLLAR/US$'],
                name="CAD/USD",
                line_color='red'))

fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['CHINA - YUAN/US$'],
                name="CNY/USD",
                line_color='blue'))


fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['EURO AREA - EURO/US$'],
                name="EUR/USD",
                line_color='orange'))

fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],
                name="GBP/USD",
                line_color='green'))


fig.update_layout(title_text="Daily Exchange Rates (2000 - 2019)")
fig.show()


# ### Daily Exchange Rates (2000 - 2009)

# #### In order to view the exchange rate behavior for the first ten years, we'll specify a specific time range. 

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
                 x=rates['Time Serie'],
                 y=rates['CANADA - CANADIAN DOLLAR/US$'],
                 name="CAD/USD",
                 line_color='red'))

fig.add_trace(go.Scatter(
                 x=rates['Time Serie'],
                 y=rates['CHINA - YUAN/US$'],
                 name="CNY/USD",
                 line_color='blue'))

fig.add_trace(go.Scatter(
                 x=rates['Time Serie'],
                 y=rates['EURO AREA - EURO/US$'],
                 name="EUR/USD",
                 line_color='orange'))

fig.add_trace(go.Scatter(
                 x=rates['Time Serie'],
                 y=rates['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],
                 name="GBP/USD",
                 line_color='green'))

fig.update_layout(xaxis_range=['2000-01-03','2009-12-31'],
                   title_text="Daily Exchange Rates (2000 - 2009)")

fig.show()


# ### Daily Exchange Rates (2010 - 2019)

# #### We'll now take a look at the exchange rate behavior for the following ten years and we'll add a range slider to navigate the time series chart. 

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['CANADA - CANADIAN DOLLAR/US$'],
                name="CAD/USD",
                line_color='red'))

fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['CHINA - YUAN/US$'],
                name="CNY/USD",
                line_color='blue'))


fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['EURO AREA - EURO/US$'],
                name="EUR/USD",
                line_color='orange'))

fig.add_trace(go.Scatter(
                x=rates['Time Serie'],
                y=rates['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],
                name="GBP/USD",
                line_color='green'))


fig.update_layout(xaxis_range=['2010-01-01','2019-12-31'],
                  title_text="Daily Exchange Rates (2010 - 2019)",
                  xaxis_rangeslider_visible=True)
fig.show()

