#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt
get_ipython().system('pip install arviz')
import arviz as az


# In[ ]:


az.style.use('arviz-darkgrid')


# **Is the week 24.12 - 31.12 an outlier in violent crime in Chicago, when accounting for temperature? Are people more peaceful during Christmas time?**
# 
# An analysis using Bayesian linear regression with PyMC3

# We use BigQuery to query a dataset from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system from 2010 to present

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.list_tables()


# Check if our query was successful:

# In[ ]:


bq_assistant.head("crime", num_rows=3)


# Check the database schema:

# In[ ]:


bq_assistant.table_schema("crime")


# Import the Chicago City Weather data:

# In[ ]:


weather = pd.read_csv("../input/historical-hourly-weather-data/temperature.csv", 
                      usecols=["datetime", "Chicago"], error_bad_lines=False)


# Check the data

# In[ ]:


weather.head()


# In[ ]:


weather.info()


# It appears there are a number of null values, let's examine them

# In[ ]:


weather[weather.isna().any(axis = 1)].head(10)


# Let's drop the rows with NaN values

# In[ ]:


weather1= weather.dropna()


# Check the date range of our weather data:

# In[ ]:


weather1['datetime'].min()


# In[ ]:


weather1['datetime'].max()


#  Now let's look at crime, specifically violent crime

# In[ ]:


violent = ['ASSAULT','BATTERY','CRIM SEXUAL ASSAULT', 'ROBBERY',
            'HOMICIDE', 'KIDNAPPING']


# Let's query the subset of crimes that are assault or battery and lead to arrests during the timeframe of the weather data:
# 
# '2012-10-01' - 
# '2017-11-30'

# In[ ]:


query1 = """SELECT
    primary_type,
    date
FROM
    `bigquery-public-data.chicago_crime.crime`
WHERE
    arrest = True 
    AND (date > '2012-10-01') AND (date < '2017-11-30')
    AND primary_type IN ('ASSAULT','BATTERY','CRIM SEXUAL ASSAULT', 'ROBBERY',
            'HOMICIDE', 'KIDNAPPING')
"""

response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head(10)


# Check data quality:

# In[ ]:


response1[response1.isna().any(axis=1)]


# Set datetime as index:

# In[ ]:


response1['datetime'] = pd.to_datetime(response1['date']).dt.tz_localize(None)
response2 = response1.set_index(['datetime'])


# In[ ]:


response2.head(3)


# Let's aggregate the crimes by count per day:

# In[ ]:


response2 = response2.resample('D').count()
response2.head()


# Aggregate average daily air temperature to match:

# In[ ]:


weather1.head(3)


# In[ ]:


weather1['datetime'] = pd.to_datetime(weather1['datetime'])
weather2 = weather1.set_index(['datetime'])
weather2 = weather2.resample('D').mean()


# In[ ]:


weather2.head(3)


# Time to combine the dataframes:

# In[ ]:


df = weather2.join(response2, on='datetime')

df.columns=['Ktemp', 'incidents', 'incidents2']


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df=df.dropna()


# Let's plot the data to get an overview:

# In[ ]:


plt.scatter(df['Ktemp'], df['incidents'])


# We see a weak correlation between temperature and incidents of violent crime, let's model this using a Bayesian linear model with pymc3:

# In[ ]:


with pm.Model() as model1: 
    alpha = pm.Normal('alpha',20,10)
    beta = pm.Normal('beta',0.2,0.1)
    sigma = pm.Uniform('sigma', 0,10)
    mu = pm.Deterministic('mu', alpha + beta* df['Ktemp'])
    pred = pm.Normal('pred', mu,sigma, observed = df['incidents'])
    trace1 = pm.sample(1000,tune = 1000, cores=2)


# Let's plot the distribution for the slope and the intercept based on our priors and the data:

# In[ ]:


az.plot_trace(trace1, var_names =['~mu'])


# Let's plot the credible interval and mean of our model:

# In[ ]:


x_points = np.linspace(250,320,100)
plt.scatter(df['Ktemp'], df['incidents'], alpha = 0.5)
plt.plot(x_points, trace1['alpha'].mean() + trace1['beta'].mean() * x_points, color = 'red')
mu_pred = trace1['alpha'] + trace1['beta'] * x_points[:,None]
prediction = stats.norm.rvs(mu_pred, trace1['sigma'])
az.plot_hpd(x_points, prediction.T, credible_interval=0.95)


# Blue dots represent incidents, the red line represents the mean incident value for given temperature. The orange represents the 0.95 credible interval. 
# 
# Now we turn our attention to the Christmas period. If separately examine the week 24.12 - 31.12, will we get the same slope and intercept?

# In[ ]:


df2 = df[(df.index.month == 12) & (df.index.day.isin([24,25,26,27,28,29,30,31]))]


# In[ ]:


df2.head()


# In[ ]:


with pm.Model() as model2: 
    alpha = pm.Normal('alpha',20,10)
    beta = pm.Normal('beta',0.2,0.1)
    sigma = pm.Uniform('sigma', 0,10)
    mu = pm.Deterministic('mu', alpha + beta* df2['Ktemp'])
    pred = pm.Normal('pred', mu,sigma, observed = df2['incidents'])
    trace2 = pm.sample(1000,tune = 1000, cores=2)


# In[ ]:


az.plot_trace(trace2, var_names=['~mu'])


# It appears that the slope is quite a bit less steep. Let's visualize the results.

# In[ ]:


x_points = np.linspace(250,290,100)
plt.scatter(df2['Ktemp'], df2['incidents'], alpha = 0.5)
plt.plot(x_points, trace2['alpha'].mean() + trace2['beta'].mean() * x_points, color = 'red')
mu_pred = trace2['alpha'] + trace2['beta'] * x_points[:,None]
prediction = stats.norm.rvs(mu_pred, trace2['sigma'])
az.plot_hpd(x_points, prediction.T, credible_interval = 0.95)


# Again, Blue dots represent incidents, the red line represents the mean incident value for given temperature. The orange represents the 0.95 credible interval. 
# 
# 
# Now let's plot the entire dataset, highlighting the results during christmas

# In[ ]:


x_points = np.linspace(250,310,100)
plt.scatter(df['Ktemp'], df['incidents'], alpha = 0.5)
plt.scatter(df2['Ktemp'], df2['incidents'], alpha = 1, color = 'red')
plt.plot(x_points, trace1['alpha'].mean() + trace1['beta'].mean() * x_points, color = 'blue')
mu_pred1 = trace1['alpha'] + trace1['beta'] * x_points[:,None]
prediction1 = stats.norm.rvs(mu_pred, trace1['sigma'])
az.plot_hpd(x_points, prediction1.T, color = 'lightblue')
x_points2 = np.linspace(254,285,100)
mu_pred2 = trace2['alpha'] + trace2['beta'] * x_points2[:,None]
prediction2 = stats.norm.rvs(mu_pred2, trace2['sigma'])
plt.plot(x_points2, trace2['alpha'].mean() + trace2['beta'].mean() * x_points, color = 'red')
az.plot_hpd(x_points2, prediction2.T, credible_interval = 0.95, color = 'red')


# The mean for the entire year is shown in blue line, while the mean for the Christmas week is shown in red line. The blue dots represent the whole year incidents, while red dots indicate incidents during the Christmas week. The 95% credible interval is shown in corresponding colors. We can see that the mean rate of violent chrime incidents is slightly lower during the Christmas week, than dates of similar temperature during other parts of the year. Thank you for reading!
