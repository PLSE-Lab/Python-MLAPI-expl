#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import rcParams
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset_Train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
dataset_Test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
dataset_Sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')


# In[ ]:


submission = dataset_Sub.head(192)

dataset_Train.fillna(0,inplace=True)
dataset_Train.describe(include='O')


# In[ ]:


dataset_Train['Date'] = pd.to_datetime(dataset_Train['Date'], infer_datetime_format = True)
gb_Train = dataset_Train.groupby("Country_Region")


# In[ ]:


confirmed_Cases = pd.pivot_table(dataset_Train, values="ConfirmedCases", index="Date", columns='Country_Region')
fatalities = pd.pivot_table(dataset_Train, values="Fatalities", index="Date", columns='Country_Region')


# In[ ]:


train_data= confirmed_Cases.loc[:,['Date', 'India']]
train_dataset = train_data
train_dataset.columns = ['ds','y']  
train_dataset['ds']= confirmed_Cases.index
prophet_basic_cnfrm = Prophet()
prophet_basic_cnfrm.fit(train_dataset)
future= prophet_basic_cnfrm.make_future_dataframe(periods=180)
forecast_cnfrm=prophet_basic_cnfrm.predict(future)
    
cnfrm = forecast_cnfrm.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['Date','Confirm']


# In[ ]:


train_data= fatalities.loc[:,['Date', 'India']]
train_dataset = train_data
train_dataset.columns = ['ds','y']  
train_dataset['ds']= fatalities.index
prophet_basic_fatal = Prophet()
prophet_basic_fatal.fit(train_dataset)
future= prophet_basic_fatal.make_future_dataframe(periods=180)
forecast_fatal=prophet_basic_fatal.predict(future)
    
fatal = forecast_fatal.loc[:,['ds','trend']]
fatal = fatal[fatal['trend']>0]
fatal.columns = ['Date','Fatalities']


# In[ ]:


fig1 =prophet_basic_cnfrm.plot(forecast_cnfrm)
fig1 =prophet_basic_fatal.plot(forecast_fatal)


# In[ ]:


fig2 = prophet_basic_cnfrm.plot_components(forecast_cnfrm)
fig2 = prophet_basic_fatal.plot_components(forecast_fatal)


# In[ ]:


from fbprophet.plot import add_changepoints_to_plot
fig = prophet_basic_cnfrm.plot(forecast_cnfrm)
a = add_changepoints_to_plot(fig.gca(), prophet_basic_cnfrm, forecast_cnfrm)


# In[ ]:


fig = prophet_basic_fatal.plot(forecast_fatal)
a = add_changepoints_to_plot(fig.gca(), prophet_basic_fatal, forecast_fatal)


# In[ ]:


prophet_basic_cnfrm.changepoints


# In[ ]:


prophet_basic_fatal.changepoints


# In[ ]:


fig = plot_plotly(prophet_basic_cnfrm, forecast_cnfrm)
py.iplot(fig) 

fig = prophet_basic_cnfrm.plot(forecast_cnfrm,xlabel='Date',ylabel='Confirmed Count')


# In[ ]:


fig = plot_plotly(prophet_basic_fatal, forecast_fatal)
py.iplot(fig) 

fig = prophet_basic_fatal.plot(forecast_fatal,xlabel='Date',ylabel='Fatalities')

