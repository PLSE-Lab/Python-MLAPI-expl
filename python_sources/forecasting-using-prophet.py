#!/usr/bin/env python
# coding: utf-8

# # Forecasting Time Series using Prophet

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


company_name = "AAPL" # Type in a company name and just run the rest


# In[ ]:


filename= "/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv".format(company_name)


# # Loading the data

# In[ ]:


df = pd.read_csv(filename)
df.head()


# In[ ]:


data = df[["date", "high"]]


# In[ ]:


data.columns = ["ds", "y"]
data.head()


# # Plotting Data

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[ ]:


data.plot()


# # Building the model
# 
# We build the model using Facebook's prophet. It makes building models very easy

# In[ ]:


from fbprophet import Prophet


# In[ ]:


m = Prophet()
m.fit(data)


# # Predicting using prophet
# 
# We will predict for the next 730 days, which leads until Febuary 7th, 2020. Increase this number even more if you want to see more to the future.

# In[ ]:


future = m.make_future_dataframe(periods=730)
future.tail()


# In[ ]:


forecast = m.predict(future)


# In[ ]:


fig = m.plot(forecast)


# In[ ]:


fig2 = m.plot_components(forecast)


# # Plotting with plotly
# We can also plot the data with plotly. It allows for interactive plots

# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()


# In[ ]:


fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# In[ ]:




