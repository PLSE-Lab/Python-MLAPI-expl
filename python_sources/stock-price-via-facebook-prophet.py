#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # ** Load Data **

# In[ ]:


data = pd.read_csv("/kaggle/input/tesla-stock-price/Tesla.csv - Tesla.csv.csv")


# # **Observing Data**

# In[ ]:


data.head()


# **Content**
# * Within the dataset one will encounter the following:
# 
# * The date - "Date"
# 
# * The opening price of the stock - "Open"
# 
# * The high price of that day - "High"
# 
# * The low price of that day - "Low"
# 
# * The closed price of that day - "Close"
# 
# * The amount of stocks traded during that day - "Volume"
# 
# * The stock's closing price that has been amended to include any distributions/corporate actions that occurs before next days open - "Adj[usted] Close"

# In[ ]:


data.describe().T


# In[ ]:


# data shape
data.shape


# In[ ]:


# data columns
data.columns


# In[ ]:


# data types
data.dtypes


# **Check Nan Values**

# In[ ]:


data.isnull().sum()


# # **Data Visualization**

# ### Disturbution Of Data

# In[ ]:


f,ax = plt.subplots(figsize = (12,7))
plt.subplot(2,1,1) 
sns.distplot(data.Open,color="green",label="Open Price");
plt.title("Open Price",fontsize = 20,color='blue')
plt.xlabel('Price',fontsize = 15,color='blue')
plt.legend()
plt.grid()
#
plt.subplot(2,1,2)
sns.distplot(data.Close,color="darkblue",label="Close Price");
plt.title("Close Price",fontsize = 20,color='blue')
plt.xlabel('Price',fontsize = 15,color='blue')
plt.tight_layout()
plt.legend()
plt.grid()


# In[ ]:


# Creating trace1
line_1 = go.Scatter(
                    x = data.index,
                    y = data.Open,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))  
data_line = [line_1]
layout = dict(title = 'TESLA Stock Price',
              xaxis= dict(title= 'Open Price',ticklen= 5,zeroline= False)
             )
fig = dict(data = data_line, layout = layout)
iplot(fig)


# # **Let's See Corrolation**

# In[ ]:


f,ax = plt.subplots(figsize = (10,7))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax);


# # **Making Prediction**

# In[ ]:


# load library
from fbprophet import Prophet


# In[ ]:


data.head()


# In[ ]:


# convert date to date:)
data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format=True)


# In[ ]:


prophet_df = data.iloc[:,[0,1]]
prophet_df.head()


# In[ ]:


prophet_df = prophet_df.rename(columns={'Date':'ds', 'Open':'y'})
prophet_df.tail(10)


# In[ ]:


prophet_df.dtypes


# In[ ]:


# Create Model
m = Prophet()
m.fit(prophet_df)


# 

# ## **Prediction For 2 years**

# In[ ]:


# Forcasting into the future
future = m.make_future_dataframe(periods=730)
future.tail(10)


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


# You can plot the forecast
figure1 = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[ ]:


# If you want to see the forecast components
figure2 = m.plot_components(forecast)


# # **Observe Prediction on Plot**

# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)

