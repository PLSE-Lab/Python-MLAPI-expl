#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Covid-19 has been a pandamic world wide. It'outbreak has been critical and during such crisis it's upto us analyse the situation and help ourselves. We take this opportunity to analyse the trend in India using Facebook's Prophet for forecasting.

# Import all the required packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from fbprophet import Prophet


# Read the data and format it accordingly (Note: Make sure to derive the dataframe columns as 'ds' that has the dates and 'y' that has the the positive cases in india till data. Prophet requires us to have the data in this format)

# In[ ]:


data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
data = pd.DataFrame(data)
data1 = data
total = data1['ConfirmedIndianNational'] + data1['ConfirmedForeignNational']
data1['y'] = total
data1.rename(columns = {'Date':'ds'}, inplace = True) 
data = data1.drop(['Sno', 'State/UnionTerritory'], axis = 1)
data1 = data1.drop(['Sno', 'State/UnionTerritory', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths'], axis = 1)
data1.head()


# Format the dates using datatime to format the dates to be able to plot using matplotlib

# In[ ]:


total_rows = np.shape(data)[0]
i = 0
dates = {}
l = []
while i != total_rows:
    if data.iloc[i,0] in dates:
        dates[str(data.iloc[i,0])]['ConfirmedIndianNational'] += data.iloc[i,1]
        dates[str(data.iloc[i,0])]['ConfirmedForeignNational'] += data.iloc[i,2]
        dates[str(data.iloc[i,0])]['Cured'] += data.iloc[i,3]
        dates[str(data.iloc[i,0])]['Deaths'] += data.iloc[i,4]
    else:
        l.append(data.iloc[i,0])
        dates[str(data.iloc[i,0])] = {
            'ConfirmedIndianNational': data.iloc[i,1],
            'ConfirmedForeignNational': data.iloc[i,2],
            'Cured': data.iloc[i,3],
            'Deaths': data.iloc[i,4]
        }
    i += 1
week_dates = [l[k:k+7] for k in range(0, len(l), 7)]
no_of_weeks = len(week_dates)
new_data = pd.DataFrame(dates)
new_data = new_data.T
total = new_data['ConfirmedIndianNational'] + new_data['ConfirmedForeignNational']
new_data = new_data.drop(['ConfirmedIndianNational', 'ConfirmedForeignNational'], axis = 1)
new_data['ConfirmedCases'] = total
new_data= new_data.T
new_data.head()
# new_data['ConfirmedCases'] = data['ConfirmedIndianNational'] + data['ConfirmedForeignNational']


# Now plot the graph, looking at the currect status of the graph the peak began to flatten a bit which could also mean the nation's lockdown might be working.

# In[ ]:


new_data
s=slice(30,40)
date = []
for i in range(len(l)):
    date.append(matplotlib.dates.date2num(datetime.strptime(l[i], '%d/%m/%y')))
ax = plt.gca()
# ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator(interval=3))
# ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%e'))
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
plt.xlabel("-----| Time |---->")
plt.ylabel("-----| Total Confirmed cases |---->")
plt.plot(date,new_data.iloc[2,:])
plt.show()


# We fit the model by instantiating a new Prophet object. Any settings to the forecasting procedure are passed into the constructor. Then you call its fit method and pass in the historical dataframe. Predictions are then made on a dataframe with a column ds containing the dates for which a prediction is to be made. You can get a suitable dataframe that extends into the future a specified number of days using the helper method Prophet.make_future_dataframe. By default it will also include the dates from the history, so we will see the model fit as well.

# In[ ]:


m = Prophet()
m.fit(data1)
future = m.make_future_dataframe(periods=7)
future.tail()


# The predict method will assign each row in future a predicted value which it names yhat. If you pass in historical dates, it will provide an in-sample fit. The forecast object here is a new dataframe that includes a column yhat with the forecast, as well as columns for components and uncertainty intervals.

# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# We plot the forecast by calling the Prophet.plot method and passing in your forecast dataframe.

# In[ ]:


fig1 = m.plot(forecast)


# We can see the forecast components, you can use the Prophet.plot_components method.

# In[ ]:


ig2 = m.plot_components(forecast)


# An interactive figure of the forecast can be created with plotly.

# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# ## Conclusion
# 
# From Prophet's plots it's evident that the given data isn't enough to forecast predicitons.
