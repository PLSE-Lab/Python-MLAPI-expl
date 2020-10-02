#!/usr/bin/env python
# coding: utf-8

# ## INTRO (acknowledgement)

# This Kernel is based on **Jose Marcial** blog post: [Link](https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c)
# 
# He's a great *data scientist* & *teacher* in the field. If you would love to learn this Kernel with more details, follow that link. You may even find his brilliant courses on udemy website: [Link](https://www.udemy.com/user/joseportilla/)
# 
# Okay, let's get started.

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Start & Process

# In[ ]:


# Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


# Read our data
data = pd.read_csv('../input/IPG2211A2N.csv',index_col=0)
data.head()


# So, it's a two-featured data frame:
# * As the data set reference, the second column is '*energy production*'.
# * The *Date* is in index now and has a string type. Then we'll also need to change its type from string to datetime

# In[ ]:


# Change our data index from string to datetime
data.index = pd.to_datetime(data.index)
data.columns = ['Energy Production']
data.head()


# The data frame is ready now!
# 
# Let's plot the data and move on to creating a proper model for our prediction.

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Plot the Data

# In[ ]:


# Import Plotly & Cufflinks libraries and run it in Offline mode
import plotly.offline as py
py.init_notebook_mode(connected=True)
py.enable_mpl_offline()

import cufflinks as cf
cf.go_offline()


# In[ ]:


# Now, plot our time serie
data.iplot(title="Energy Production Between Jan 1939 to May 2019")


# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Decomposition

# We'll need to *deconstruct* our time series into several components: **Seasonal**, **Trend** and **Residual**. There are a few ways to do that, but the most easy one is using **statsmodels**.

# In[ ]:


# We'll use statsmodels to perform a decomposition of this time series
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model='multiplicative')

fig = result.plot()


# Plot a better decompositioned charts using Plotly:

# In[ ]:


py.iplot_mpl(fig)
# Try "py.plot_mpl(fig)" on your local Anaconda, it'll show greater plot than this one


# The plot itself shows us that we have a 'seasonal' component in our time series, which means we need to choose p,d,q for ARIMA and P,D,Q for SARIMA

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Seasonal ARIMA (SARIMA) & Grid Search

# #### Pmdarima library is not available in Kaggle by default, so we'll need to install it for this Kernel

# In[ ]:


get_ipython().system('pip install pmdarima')


# In[ ]:


# Kaggle doesn't have the up-to-date version of Scipy, so we need to upgrade it in order to use pmdarima library
get_ipython().system('pip install --upgrade scipy')


# Let's do the **grid search** now!

# In[ ]:


# The Pmdarima library for Python allows us to quickly perform this grid search 
from pmdarima import auto_arima


# A few hints about auto_arima method:
# 
# * **trace**: *bool* / whether to print status on the fits (**I'd like to see what's actually happening behind the code, so I enable it. You can eliminate this parameter to skip these lines and save your time**)
# * **error_action**: *str* / warn or raise or ignore , if unabe to fit an ARIMA due to stationarity issue
# * **suppress_warnings**: *bool* / Many warnings might be thrown inside of statsmodels, so we'll try to squelch them all.
# * **stepwise**: *bool* / The algorithm can be significantly faster than fitting all hyper-parameter combinations and is less likely to over-fit the mode.

# In[ ]:


stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)


# The **AIC** (*Akaike information criterion*) is an **estimator** of the relative quality of statistical models for a given set of data. AIC estimates the quality of each model, relative to each of the other models.
# 
# The **AIC** value allow us to **compare** how well a model fits the data. Models that have a better fit while using fewer features, will receive a better AIC score (*lower*) than similar models that utilize more features.

# In[ ]:


print(stepwise_model.aic())


# Great! we can fit the model now

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Train Test Split

# * For the **Test**: we'll need to chop off a portion of our latest data, say from 2016, Jan.
# * Fore the **Train**: we'll train on the rest of the data after splitting up the test portion.

# In[ ]:


# For the Test: we'll need to chop off a portion of our latest data, say from 2016, Jan.
test = data.loc['2016-01-01':]

# Fore the Train: we'll train on the rest of the data after split the test portion
train = data.loc['1939-01-01':'2015-12-01']


# ## Train Model

# In[ ]:


stepwise_model.fit(train)


# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Evaluation

# From 2016-01-01 to 2019-05-01 (*the latest update from the source*), we have 41 rows:

# In[ ]:


future_forecast = stepwise_model.predict(n_periods=41)
print(future_forecast)


# * **Create** a data frame contains our prediction (future_forecast):

# In[ ]:


future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])


# * **Concatenate** *future_forecast* and *test data frame* together and plot the result:

# In[ ]:


pd.concat([test, future_forecast], axis=1).iplot()


# * **Compare** *future_forecast* with the *entire data set* to get a larger picture of the context of our prediction:

# In[ ]:


pd.concat([data,future_forecast],axis=1).iplot()


# Okay, we have **evaluated** our model and it seems quite acceptable.
# 
# * It's time to refit our model in the entire data set and then forecast the real future!

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ### Forecasting example for the next year

# In[ ]:


stepwise_model.fit(data)


# For instance, let's forecast the "energy production" for the next year:

# In[ ]:


future_forecast_1year = stepwise_model.predict(n_periods=13)


# In[ ]:


# For a year forecasting, we need 13 rows from 2019-05-01 to 2020-05-01
next_year = [pd.to_datetime('2019-05-01'),
            pd.to_datetime('2019-06-01'),
            pd.to_datetime('2019-07-01'),
            pd.to_datetime('2019-08-01'),
            pd.to_datetime('2019-09-01'),
            pd.to_datetime('2019-10-01'),
            pd.to_datetime('2019-11-01'),
            pd.to_datetime('2019-12-01'),
            pd.to_datetime('2020-01-01'),
            pd.to_datetime('2020-02-01'),
            pd.to_datetime('2020-03-01'),
            pd.to_datetime('2020-04-01'),
            pd.to_datetime('2020-05-01')]


# In[ ]:


future_forecast_1year = pd.DataFrame(future_forecast_1year, index=next_year, columns=['Prediction'])


# In[ ]:


pd.concat([data,future_forecast_1year],axis=1).iplot()


# Seems okay to me, right? :)

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)
