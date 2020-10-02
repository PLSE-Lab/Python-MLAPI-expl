#!/usr/bin/env python
# coding: utf-8

# # Motivation
# In this kernel I explore some stuff tangentially related to this competition, but I slightly reframe the problem. Instead of trying to detect specific cases of fraud based on transaction characteristics I will look at trying to determine the amount of fraud on a given day or what percentage of transactions will be fradulent on a given day. This would be useful for a credit card company in order to adequetely prepare their fraud departments for the busiest times of year and may also influence their fraud models in order to be more diligent during these times of the year. 
# 
# In this kernel I will be using FB Prophet, a very useful and simple tool for time-series forecasting. If you'd like to read about the math and theory behind this tool you can check it out at: https://peerj.com/preprints/3190/. At the heart of this tool is a [generalized additive model](https://www.youtube.com/watch?v=f9Rj6SHPHUU). I will focus on the basic usage of prophet and what insights we can pull out of this tool

# In order to make this work I needed to do one small tweak I found on github since the plotting was not working. There is some conflict between newer versions of pandas and plotting libraries and prophet. The trick is to call pd.plotting.register_matplotlib_converters(). I found this trick at https://github.com/facebook/prophet/issues/999

# ## Installing and importing prophet

# In[ ]:


get_ipython().system('pip install -U fbprophet')
from fbprophet import Prophet


# In[ ]:


import numpy as np
import pandas as pd
import datetime
import math
import gc
pd.plotting.register_matplotlib_converters()

PATH = '../input/'
START_DATE = '2017-12-01'


# ## Loading in the data

# In[ ]:


# Load data
train = pd.read_csv(PATH + 'train_transaction.csv')
test = pd.read_csv(PATH + 'test_transaction.csv')

df = pd.concat([train, test], axis = 0, sort = False)
del train, test
gc.collect()


# ## Converting the datetime field to match localized date and time

# In[ ]:


# Preprocess date column
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

print(df['TransactionDT'].head())
print(df['TransactionDT'].tail())


# ## Looking at number of fraud cases per day

# In[ ]:


tm_df = df.groupby(by=df['TransactionDT'].dt.date)["isFraud"].count()


# In[ ]:


tm_df = tm_df.reset_index()


# In[ ]:


n_train = 182
n_test = 365 - n_train


# ## Formatting data for Prophet
# Prophet is specifically looking for a ds and y column. ds will obviously be the date we previously converted and the y will be whatever we want to forecast, in this case it is the count of fraud on that day

# In[ ]:


tm_df.columns = ["ds", "y"]
tm_df


# ## Plotting the count of fraud per day
# We can see two fairly sizeable peaks 

# In[ ]:


tm_df.iloc[:n_train].plot(x = "ds", y = "y")


# ## Fitting the model
# Prophet is very convenient once you get the data in the correct format. All we have to do is instantiate and then fit. Similar to sklearn once you get the correct format. We will specifically train against the train set and then forecast against the test set

# In[ ]:


m = Prophet()
m.fit(tm_df.iloc[:n_train])


# ## Making future dataframe
# Now Prophet will help us by extending our dataframe beyond it's previously known dates so we can define how long we want to forecast into the future (measured in days). We will forecast out the rest of the year

# In[ ]:


future = m.make_future_dataframe(periods=n_test)
future.tail()


# ## Making our forecast
# Now Prophet has been fit we can make predictions into the future. Let's see what the crystal ball says.

# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig1 = m.plot(forecast)


# ## Visualizing predictions
# Prophet will output several things in it's predictions. It will output yhat (it's forecast), yhat_lower(the lower bound it expects), yhat_upper(the upper bound it expects). These function as a sort of confidence interval. 
# 
# Right now our forecast seems to think fraud is eventually decaying away, but we know this likely isn't the case. It seems to be cause the high outliers ocurring in a few different places. We can do a couple things here to try to clean up the predictions, but I will touch on those again a little later. Let's look a bit more at what prophet can do for us in terms of visualization

# In[ ]:


forecast


# ## Explainability and Seasonal Decomposition
# One of the major advatnages of prophet is it's focus on explainability. It generates various different outputs that allow us additional introspection of prophets view of the trends and patterns in the data

# In[ ]:


fig2 = m.plot_components(forecast)


# What we see here is the trend across the entire year. The dark blue line is the trend it fit across the year. We can see that this is steeply declining like we saw in the previous graph. The lighter boundary above and below it is the confidence it has at it reaches further in the future. You can see that as it is looking further it has less confidence and expects more variation
# 
# In the second graph we see the daily trends. This is showing us the relative values per day of the week. We can see that saturday seems to be the highest day for fraud at least in the eyes of prophet. 

# ## Interactive plots with plotly
# 
# Prophet also has good integration with plotly in order to create easy interactive plots

# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# ## Fixing the spurious negative trend
# Now we can look at the various ways we can remove that negative trend that is being heavily effected by outliers. In the prophet docs they mention that outliers should simply be dropped, but I'll try to be a bit more creative than that as those days are likely very important to us. 
# 
# One of the nice things about prophet is it can show us where it believes changes in a trend have ocurred and we can change how aggresive it is in finding changepoints. We can try some values and hope that we can make the model find that those points at the beginning were changepoints and anything beyond them should follow a different trend

# In[ ]:


m = Prophet(changepoint_prior_scale=0.3)
forecast = m.fit(tm_df.iloc[:n_train]).predict(future)
fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# In[ ]:


from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# After some guess and check we can see that increasing the changepoint_prior_scale to .3 it seems to pretty accurately capture the various points where trends change and removes the strong decrease in our future forecasts. It is still slightly sloping downward. We don't expect our forecast to be entirely stationary, but there is likely a better way to handle this. Increasing the changepoint_prior_scale causes our predictions to be very periodic and our uncertainty in the future to be exceptionally high. 

# Alternately we can manually define changepoints. Let's try to find some. 

# In[ ]:


#let's pick the 3 biggest values and use them as changepoints
changepoints = tm_df.iloc[:n_train].nlargest(3, "y")["ds"]


# In[ ]:


m = Prophet(changepoints=changepoints)
forecast = m.fit(tm_df.iloc[:n_train]).predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# It seems to do ok, but not great. It doesnt seem to fully remove the negative trend created by the outliers. One last attempt before we just drop the outliers all together

# ## Adding in holidays
# Now we can try adding in holidays. Prophet has special provisions in order to try to acknowledge holidays in slightly different ways. 

# In[ ]:


m = Prophet()
m.add_country_holidays(country_name='US')
forecast = m.fit(tm_df.iloc[:n_train]).predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# Looks like it is able to capture the first spike in fraud better than any other method we have seen so far, but it still doesnt seem to be capturing the second outlier point. Maybe we can add this in as another manually defined holiday

# In[ ]:


manual_holidays = pd.DataFrame({
  'holiday': 'manual',
  'ds': pd.to_datetime(['2018-03-3']),
  'lower_window': -1,
  'upper_window': 1,
})

m = Prophet(holidays=manual_holidays, changepoint_prior_scale=.05)
m.add_country_holidays(country_name='US')
forecast = m.fit(tm_df.iloc[:n_train]).predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# Looks like we are able to capture the major outliers with this holiday stuff but we still see a downward trend. This likely would be fixed if we had more than half a years data because we would learn more about the nuances across the entire year, but we dont have that luxury. 

# ## Remove the outliers
# 
# One simple way we can determine which points to remove is taking the mean and std and remove points 3 std away from the mean. 

# In[ ]:


train_std = tm_df.iloc[:n_train]["y"].std()
train_mean = tm_df.iloc[:n_train]["y"].mean()


# We'll filter values above 2.5 std from the mean and set them to nan so they dont effect our model at all

# In[ ]:


tm_df.loc[(tm_df["y"] > (train_mean + 2.5*train_std)), "y"] = np.nan


# In[ ]:


m = Prophet()
forecast = m.fit(tm_df.iloc[:n_train]).predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# Hmm even that doesn't seem to fix the strongly negative trend. Maybe if we approach the problem slightly differently and look at the percentage of transactions that are fradulent we will be able to model that more successfully

# In[ ]:


tm_df = df.groupby(by=df['TransactionDT'].dt.date)["isFraud"].mean()
tm_df = tm_df.reset_index()
tm_df.columns = ["ds", "y"]


# In[ ]:


m = Prophet(changepoint_prior_scale=.05)
forecast = m.fit(tm_df.iloc[:n_train]).predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# Interestingly we can see that the changepoints fall on a similar section and we have ranges from 1 percent of transactions being fraudlent and 7 percent. That can have serious business implications

# In[ ]:




