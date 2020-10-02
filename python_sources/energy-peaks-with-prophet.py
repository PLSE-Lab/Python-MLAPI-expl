#!/usr/bin/env python
# coding: utf-8

# **Analysis and Modelling on Energy Consumption of Duquesne Light Co. (DUQ)**

# In[ ]:


from IPython.display import Image
Image("../input/image-power/power-grid-orig.jpg")


# **Note:**  
# Kindly upvote the kernel if you find it useful. Suggestions are always welome. Let me know your thoughts in the comment if any.

# **Hourly Energy Consumption of Duquesne Light Co. (DUQ)**  
# For the Exploratory analysis and Model building, I will be using Duquesne Light Co. (DUQ) dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# **Reading the Data**  

# In[ ]:


#Reading the Datafile
duq_dt = pd.read_csv('../input/hourly-energy-consumption/DUQ_hourly.csv', parse_dates=True)

#Extracting Year, Month, Date, Day, Week and Hour details
duq_dt['Datetime'] =  pd.to_datetime(duq_dt['Datetime'])
duq_dt['Year'] = duq_dt['Datetime'].dt.year
duq_dt['Month'] = duq_dt['Datetime'].dt.month
duq_dt['Day'] = duq_dt['Datetime'].dt.day
duq_dt['Week'] = duq_dt['Datetime'].dt.dayofweek
duq_dt['Hour'] = duq_dt['Datetime'].dt.hour
duq_dt['DName'] = duq_dt['Datetime'].dt.day_name()


# **Plottting the Energy Trend **

# In[ ]:


#Plotting the Energy Trend
Eng_trend = duq_dt.iloc[:,0:2]

import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='pearl')

Eng_trend.iplot(x='Datetime',kind='scatter', fill = False, 
                title = "Energy Trend of Duquesne Light Co. (DUQ) from 2005 - 2018 in MW")


# In[ ]:


range_dt = go.Scatter(
    x=Eng_trend.Datetime,
    y=Eng_trend['DUQ_MW'],
    line = dict(color = '#2a932e'),
    opacity = 0.6)

data = [range_dt]

layout = dict(
    title='Energy Trend of Duquesne Light Co. (DUQ) from 2005 - 2018 in MW with Slider',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(count=3,
                     label='3y',
                     step='year',
                     stepmode='backward'),
                dict(count=6,
                     label='6y',
                     step='year',
                     stepmode='backward'),
                dict(count=9,
                     label='9y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Time Series with Rangeslider")


# From the above plot its clearly seen that there has been a repetative pattern over the period of time. But, In order to uncover the pattern the data needs to be analysed to atmost granular level. Therefore, we will analyse the data at different granular level with respect to month, day and hour details.

# **Analysis Interval**  
# For analysis purpose, I will be considering 6 years of data from 2012 to 2017.

# In[ ]:


#Data Filtering 2012 - 2017
duq_dt_5 = duq_dt[(duq_dt.Year >= 2012) & (duq_dt.Year <= 2017)]


# **Average Energy Consumption w.r.t Year and Month**

# In[ ]:


#Calculating average energy consumption w.r.t to Year and Month
yr_mon = duq_dt_5.groupby(['Year', 'Month'], as_index = False)['DUQ_MW'].mean()

trace_2012 = go.Scatter(
                x=yr_mon.Month[yr_mon.Year == 2012],
                y=yr_mon.DUQ_MW[yr_mon.Year == 2012],
                name = "2012",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace_2013 = go.Scatter(
                x=yr_mon.Month[yr_mon.Year == 2013],
                y=yr_mon.DUQ_MW[yr_mon.Year == 2013],
                name = "2013",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

trace_2014 = go.Scatter(
                x=yr_mon.Month[yr_mon.Year == 2014],
                y=yr_mon.DUQ_MW[yr_mon.Year == 2014],
                name = "2014",
                line = dict(color = '#29a944'),
                opacity = 0.8)

trace_2015 = go.Scatter(
                x=yr_mon.Month[yr_mon.Year == 2015],
                y=yr_mon.DUQ_MW[yr_mon.Year == 2015],
                name = "2015",
                line = dict(color = '#279ff0'),
                opacity = 0.8)

trace_2016 = go.Scatter(
                x=yr_mon.Month[yr_mon.Year == 2016],
                y=yr_mon.DUQ_MW[yr_mon.Year == 2016],
                name = "2016",
                line = dict(color = '#db9a08'),
                opacity = 0.8)

trace_2017 = go.Scatter(
                x=yr_mon.Month[yr_mon.Year == 2017],
                y=yr_mon.DUQ_MW[yr_mon.Year == 2017],
                name = "2017",
                line = dict(color = '#c90645'),
                opacity = 0.8)

data = [trace_2012, trace_2013, trace_2014, trace_2015, trace_2016, trace_2017]

layout = dict(
    xaxis=dict(title='Numerical Month'),
    yaxis=dict(title='Average Energy Consumption in MW'),
    title = "Month Wise Average Engergy Consumption [2012 - 2017]",
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Month")


# For the year 2012, 2013 and 2017 the peak month of average energy consumption seems to be on month 7 which is July. Further, for all the years from 2012 to 2015 there are two sharp drops on the 4th and 10th month which are April and October respectively.

# **Average Energy Consumption w.r.t Year and Days**

# In[ ]:


#Calculating average energy consumption w.r.t to Year and Days
yr_days = duq_dt_5.groupby(['Year', 'Day'], as_index = False)['DUQ_MW'].mean()

trace_2012 = go.Scatter(
                x=yr_days.Day[yr_days.Year == 2012],
                y=yr_days.DUQ_MW[yr_days.Year == 2012],
                name = "2012",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace_2013 = go.Scatter(
                x=yr_days.Day[yr_days.Year == 2013],
                y=yr_days.DUQ_MW[yr_days.Year == 2013],
                name = "2013",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

trace_2014 = go.Scatter(
                x=yr_days.Day[yr_days.Year == 2014],
                y=yr_days.DUQ_MW[yr_days.Year == 2014],
                name = "2014",
                line = dict(color = '#29a944'),
                opacity = 0.8)

trace_2015 = go.Scatter(
                x=yr_days.Day[yr_days.Year == 2015],
                y=yr_days.DUQ_MW[yr_days.Year == 2015],
                name = "2015",
                line = dict(color = '#279ff0'),
                opacity = 0.8)

trace_2016 = go.Scatter(
                x=yr_days.Day[yr_days.Year == 2016],
                y=yr_days.DUQ_MW[yr_days.Year == 2016],
                name = "2016",
                line = dict(color = '#db9a08'),
                opacity = 0.8)

trace_2017 = go.Scatter(
                x=yr_days.Day[yr_days.Year == 2017],
                y=yr_days.DUQ_MW[yr_days.Year == 2017],
                name = "2017",
                line = dict(color = '#c90645'),
                opacity = 0.8)

data = [trace_2012, trace_2013, trace_2014, trace_2015, trace_2016, trace_2017]

layout = dict(
    xaxis=dict(title='Day of the Month'),
    yaxis=dict(title='Average Energy Consumption in MW'),
    title = "Day Wise Average Engergy Consumption [2012 - 2017]",
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Days")


# Every year from 2012 to 2017, between 12th to 17th and 23rd to 26 there has been a decreasing trend in average enery consumption.

# **Average Energy Consumption w.r.t Year and Week Day**

# In[ ]:


#Calculating average energy consumption w.r.t to Year and Days
yr_wdays = duq_dt_5.groupby(['Year', 'Week', 'DName'], as_index = False)['DUQ_MW'].mean()

trace_2012 = go.Scatter(
                x=yr_wdays.DName[yr_wdays.Year == 2012],
                y=yr_wdays.DUQ_MW[yr_wdays.Year == 2012],
                name = "2012",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace_2013 = go.Scatter(
                x=yr_wdays.DName[yr_wdays.Year == 2013],
                y=yr_wdays.DUQ_MW[yr_wdays.Year == 2013],
                name = "2013",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

trace_2014 = go.Scatter(
                x=yr_wdays.DName[yr_wdays.Year == 2014],
                y=yr_wdays.DUQ_MW[yr_wdays.Year == 2014],
                name = "2014",
                line = dict(color = '#29a944'),
                opacity = 0.8)

trace_2015 = go.Scatter(
                x=yr_wdays.DName[yr_wdays.Year == 2015],
                y=yr_wdays.DUQ_MW[yr_wdays.Year == 2015],
                name = "2015",
                line = dict(color = '#279ff0'),
                opacity = 0.8)

trace_2016 = go.Scatter(
                x=yr_wdays.DName[yr_wdays.Year == 2016],
                y=yr_wdays.DUQ_MW[yr_wdays.Year == 2016],
                name = "2016",
                line = dict(color = '#db9a08'),
                opacity = 0.8)

trace_2017 = go.Scatter(
                x=yr_wdays.DName[yr_wdays.Year == 2017],
                y=yr_wdays.DUQ_MW[yr_wdays.Year == 2017],
                name = "2017",
                line = dict(color = '#c90645'),
                opacity = 0.8)

data = [trace_2012, trace_2013, trace_2014, trace_2015, trace_2016, trace_2017]

layout = dict(
    xaxis=dict(title='Day of the Week'),
    yaxis=dict(title='Average Energy Consumption in MW'),
    title = "Day of the week Wise Average Engergy Consumption [2012 - 2017]",
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Week")


# For all the years from 2012 to 2017, the average energy consumption is up and running from Monday to Friday and drops during the weekends. So, Over all the average energy consumption is starting high on the start of the weekday and ends in a downward trend during the weekends.  
# One good thing about this plot is that, we can clearly see that the average energy utilization has significantly reduced over the period from 2012 to 2017.

# **Average Energy Consumption w.r.t Year and Hour**

# In[ ]:


#Calculating average energy consumption w.r.t to Year and Hour
yr_hr = duq_dt_5.groupby(['Year', 'Hour'], as_index = False)['DUQ_MW'].mean()

trace_2012 = go.Scatter(
                x=yr_hr.Hour[yr_hr.Year == 2012],
                y=yr_hr.DUQ_MW[yr_hr.Year == 2012],
                name = "2012",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace_2013 = go.Scatter(
                x=yr_hr.Hour[yr_hr.Year == 2013],
                y=yr_hr.DUQ_MW[yr_hr.Year == 2013],
                name = "2013",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

trace_2014 = go.Scatter(
                x=yr_hr.Hour[yr_hr.Year == 2014],
                y=yr_hr.DUQ_MW[yr_hr.Year == 2014],
                name = "2014",
                line = dict(color = '#29a944'),
                opacity = 0.8)

trace_2015 = go.Scatter(
                x=yr_hr.Hour[yr_hr.Year == 2015],
                y=yr_hr.DUQ_MW[yr_hr.Year == 2015],
                name = "2015",
                line = dict(color = '#279ff0'),
                opacity = 0.8)

trace_2016 = go.Scatter(
                x=yr_hr.Hour[yr_hr.Year == 2016],
                y=yr_hr.DUQ_MW[yr_hr.Year == 2016],
                name = "2016",
                line = dict(color = '#db9a08'),
                opacity = 0.8)

trace_2017 = go.Scatter(
                x=yr_hr.Hour[yr_hr.Year == 2017],
                y=yr_hr.DUQ_MW[yr_hr.Year == 2017],
                name = "2017",
                line = dict(color = '#c90645'),
                opacity = 0.8)

data = [trace_2012, trace_2013, trace_2014, trace_2015, trace_2016, trace_2017]

layout = dict(
    xaxis=dict(title='Time of the Day'),
    yaxis=dict(title='Average Energy Consumption in MW'),
    title = "Time Wise Average Engergy Consumption [2012 - 2017]",
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Time")


# We can clearly observe that the day starts with little high consumption of energy compared to to early morning hours around 4:00 AM. After 5:00 AM the energy consumption starts to rise till 6:00 PM in the evening and then gradully starts to decrease towards the end of the day. 

# **Modelling using Prophet from Facebook**

# In[ ]:


#Reading the dataset
duq_dt = pd.read_csv('../input/hourly-energy-consumption/DUQ_hourly.csv', index_col=[0], parse_dates=[0])


# **Train - Test Split**  
# For model building, first we need the training and testing datasets. Here, we are going to split the data with respect to date column in the dataset.   
# * Training data -->  From 1-Jan-2008 to 31-Dec-2015  
# * Testing data --->  From 1-Jan-2016 to 31-Jul-2018

# In[ ]:


#Train & Test data based on date index
split_index = '01-Jan-2015'
duq_dt_tr = duq_dt.loc[duq_dt.index <= split_index].copy()
duq_dt_ts = duq_dt.loc[duq_dt.index > split_index].copy()


# **Visualizing the training and test dataset**

# In[ ]:


#Visualizing the training and test dataset
_ = duq_dt_ts     .rename(columns={'DUQ_MW': 'TEST SET'})     .join(duq_dt_tr.rename(columns={'DUQ_MW': 'TRAIN SET'}), how='outer')     .plot(figsize=(15,5), title='Duquesne Light Co. (DUQ)', style='.')


# **Formatting the data for modelling**

# In[ ]:


# Formatting the training data for prophet model using ds and y
duq_dt_tr.reset_index().rename(columns={'Datetime':'ds', 'DUQ_MW':'y'}).head()


# **Setting up the model for Training**

# In[ ]:


# Setup and train model
from fbprophet import Prophet
model = Prophet()
model.fit(duq_dt_tr.reset_index().rename(columns={'Datetime':'ds', 'DUQ_MW':'y'}))


# **Predict on testing set with model**

# In[ ]:


# Predict on training set with model
duq_dt_ts_forecast = model.predict(df=duq_dt_ts.reset_index().rename(columns={'Datetime':'ds'}))


# **Plotting the forecast**

# In[ ]:


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(duq_dt_ts_forecast, ax=ax)


# **Plotting the components of the forecasted model**

# In[ ]:


# Plot the components
fig = model.plot_components(duq_dt_ts_forecast)


# **Compare Forecast to Actuals**
# * Green - Actuals
# * Blue - Forceast

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(duq_dt_ts.index, duq_dt_ts['DUQ_MW'], color='g')
fig = model.plot(duq_dt_ts_forecast, ax=ax)


# **Single Month Prediction**

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(duq_dt_ts.index, duq_dt_ts['DUQ_MW'], color='g')
fig = model.plot(duq_dt_ts_forecast, ax=ax)
ax.set_xbound(lower='01-01-2016', upper='02-01-2016')
ax.set_ylim(0, 3000)
plot = plt.suptitle('January 2017 Forecast vs Actuals')


# **Single Week of Predictions**

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(duq_dt_ts.index, duq_dt_ts['DUQ_MW'], color='g')
fig = model.plot(duq_dt_ts_forecast, ax=ax)
ax.set_xbound(lower='01-01-2016', upper='01-08-2016')
ax.set_ylim(0, 3000)
plot = plt.suptitle('First Week of January Forecast vs Actuals')


# **Error Metrics**

# * Mean Square Error

# In[ ]:


mean_squared_error(y_true=duq_dt_ts['DUQ_MW'],
                   y_pred=duq_dt_ts_forecast['yhat'])


# * Mean Absolute Error

# In[ ]:


mean_absolute_error(y_true=duq_dt_ts['DUQ_MW'],
                   y_pred=duq_dt_ts_forecast['yhat'])


# * Mean absolute percentage error [MAPE]

# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=duq_dt_ts['DUQ_MW'],
                   y_pred=duq_dt_ts_forecast['yhat'])


# **Impact of Holidays**  
# As of now the model is implemented without considering the holidays. Now, Let us ingest the holiday parameter into the model and see how the model is impacted.
# 
# Prophet comes with a Holiday Effects parameter that can be provided to the model prior to training.
# 
# We will use the built in  ** pandas USFederalHolidayCalendar** to pull the list of holidays.

# In[ ]:


#Calandar details
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()
tr_holidays = cal.holidays(start=duq_dt_tr.index.min(), end=duq_dt_tr.index.max())
ts_holidays = cal.holidays(start=duq_dt_ts.index.min(), end=duq_dt_ts.index.max())


# In[ ]:


# Create a dataframe with ds columns and Holiday
duq_dt['date'] = duq_dt.index.date
duq_dt['is_holiday'] = duq_dt.date.isin([d.date() for d in cal.holidays()])
holiday_df = duq_dt.loc[duq_dt['is_holiday']].reset_index().rename(columns={'Datetime':'ds'})
holiday_df['holiday'] = 'USFederalHoliday'
holiday_df = holiday_df.drop(['DUQ_MW','date','is_holiday'], axis=1)


# **Setting up the Model with holidays**

# In[ ]:


# Setup and train model with holidays
model_with_holidays = Prophet(holidays=holiday_df)
model_with_holidays.fit(duq_dt_tr.reset_index().rename(columns={'Datetime':'ds', 'DUQ_MW':'y'}))


# **Effect of Holiday on the Model components**

# In[ ]:


fig2 = model_with_holidays.plot_components(duq_dt_ts_forecast)


# **Model Prediction with Holidays**

# In[ ]:


# Predict on training set with model
duq_test_fcst_with_hols = model_with_holidays.predict(df=duq_dt_ts.reset_index().rename(columns={'Datetime':'ds'}))

duq_test_fcst_with_hols.head()

