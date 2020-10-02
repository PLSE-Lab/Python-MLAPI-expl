#!/usr/bin/env python
# coding: utf-8

# # Divvy Bicycle Sharing Weather Analysis

# In this notebook we'll be exploring the effect of weather on bicycle sharing volume as provided by Divvy. The data is available on [Kaggle](https://www.kaggle.com/yingwurenjian/chicago-divvy-bicycle-sharing-data). 
# 
# The analysis approach is as follows:
# - **Preparation:** ensuring there are no missing values and that we're familiar with the data that is presented and their representations.
# 
# 
# - **Data Manipulation:**
#     1. Create summarized views of the data at reasonable granularities, which were daily and hourly
#     2. Add in flags for special events such as weekends, holidays, and holiday weekends
#     
#     
# - **Exploratory Data Analysis:** quick views of the data to see general trends on both average trip time and trip counts; get a sense of some of the major correlations, especially for weather variables
# 
# 
# - **Decomposing Trend / Seasonality:** this is done to try and remove the seasonality and trend components from the time series to isolate the residuals which could then be modeled with just the weather variables
# 
# 
# - **Regression Analysis:** linear regressions are run on both un-decomposed data (with dummies for time variables) and the decomposed data; results are then compiled and summarized for the weather variables
# 
# ## Key Takeaways
# - **Warm Temperature:** 
#     - Bike volume tend to increase when it's quite warm (between 60 and 80 degrees); however, as temperature increases beyond 80, the volume drops off as it's perhaps too hot out to be biking; the optimal range is about 70-80 degrees
#     - Average trip duration increases with temperature as well; it has very high correlation with trip count and duration (~50% for both)
#     - Temperature is also correlated to the number of months / annual seasonality which is why in the regression analysis it was important to separate out the effect of seasonality to get a sense of how important is temperature in excess of what is expected at that time of the year
#     
#     
# - **Weather Events:** Cloudy days sees the most disproportionately high number of trips (80% of the time it's cloudy yet 85% of trips are on cloudy days); clear and snow / rain sees a disproportionately low number of trips; weather events doesn't have a material effect on trip duration except for snow / rain which decreases the average ride time by up to a minute. 
# 
# 
# - **Weekend vs. Weekdays:** weekdays show a high amount of volume for commuters and the daily ride volume is about ~40% on weekdays than on weekends. Also the rides on weekdays are more concentrated around rush hour (9am and 6pm). However, the average ride times are higher on the weekends as they are more likely for leisure. 
# 
# 
# - **Regression Coefficients** are summarized in the table below; they only include coefficients that were tested to be statistically significant (p-value < 0.05). There are two levels to the columns: 1) the data_type which specifies whether the original (un-decomposed) data was used or the residuals from the trend/seasonality decomposition; and 2) which specifies whether the data was aggregated at the daily level or the hourly level. Some interesting insights from here are:
#     - Generally weather events do not have a huge impact on average duration, however we do see that in rain or snow, the trip lengths decrease about 0.5-1min on average; similar story for temperature - hotter -> longer rides but not by much
#     - For trip counts we do see generally thunderstorms have the biggest impact on rental volume (decrease of 100+ rentals per hour) whereas the other events are similar but there isn't a clear ordering or significance
#     - The residuals showed much fewer weather variables of significance than the original data (with dummies); this is likely because the decomposition algorithm captures a lot of the weather patterns as part of the cyclicality and so the leftover variance is difficult to be explained by weather -- it could only be explained by unexpected weather events

# In[ ]:


# pivot_results


# ## Future Analyses
# It would be interesting to see how much the effect of weather differs by the time of year and types of days. For example, it could be that an unexpected weather event on a weekend holiday would be much more detrimental to trip volume than an unexpected event during a winter weekday. To test that, we could separate the data by the type of day or into months and run similar cuts and regression analyses to see the weather impact.

# ## Table of Contents
# 0. [Preparation](#Preparation)
# 1. [Data Manipulation](#Data-Manipulation)
# 2. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# 3. [Decomposing Trend and Seasonality](#Decomposing-Trend-and-Seasonality)
# 4. [Regression Analysis](#Regression-Analysis)

# ## Preparation

# In[ ]:


# Import libraries for data manipulation
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

import seaborn as sns

# Import library for modeling
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# In[ ]:


# Read in csv file
df_in = pd.read_csv('../input/data.csv')


# In[ ]:


# Peek at data
df_in.head()


# In[ ]:


# Get general description of numerical fields in input dataframe
desc = df_in.describe()
desc.apply(lambda x: x.apply(lambda y: "{:.1f}".format(y)))


# In[ ]:


# Get general information on input dataframe
df_in.info()


# In[ ]:


# Locate any nulls in the dataset
df_in.isnull().sum(0)


# ## Data Manipulation

# ### Level of Granularity
# We need to determine the level of granularity that further analysis should be done at. There are two factors that will influence this decision:
# 1. At what level will the data have a good enough signal without losing too much fidelity, that is, we don't want the data to be too sparse and spikey, at the same time, we want the grouping to conceal the effects of some of the factors 
# 2. What level is the weather data at? Are the temperature and events fields at a very granular level, and if we wanted to group it, how do we summarize those fields?

# In[ ]:


# Determine the granularity of the weather data

# Is it daily?
daily = df_in[['year', 'month', 'week', 'day', 'events', 'temperature']].drop_duplicates().shape[0]             == df_in[['year', 'month', 'week', 'day']].drop_duplicates().shape[0]

# Is it hourly?
hourly = df_in[['year', 'month', 'week', 'day', 'hour', 'events', 'temperature']].drop_duplicates().shape[0]             == df_in[['year', 'month', 'week', 'day', 'hour']].drop_duplicates().shape[0]
print("Daily?", daily, "; Hourly?", hourly)


# Since the weather data is at the hourly level, we'll summarize the dataset by the hour but also keep a daily level dataframe as comparison. To summarize weather event and temperature at the daily level, we'll use mode and mean respectively. 

# In[ ]:


# Add date column for starting date
df_in['start_date'] = df_in['starttime'].str[:10]
df_in = df_in.sort_values(by=['starttime']).reset_index(drop=True)


# In[ ]:


# Create new summarized dataset at the hourly level
df_in['tripcount'] = 1
df_hourly = df_in.groupby(['start_date', 'year', 'month', 'week', 'day', 'hour', 'temperature', 'events'])                  .agg({'tripcount':'sum', 'tripduration':'sum'}).reset_index()
df_hourly['avg_duration'] = df_hourly['tripduration'] / df_hourly['tripcount']
df_hourly.head()


# In[ ]:


# Create daily level dataset
df_daily = df_hourly.groupby(['start_date', 'year', 'month', 'week', 'day'])                     .agg({'temperature':'mean', 'events':lambda x:x.value_counts().index[0],
                          'tripcount':'sum', 'tripduration':'sum'}).reset_index()
    
df_daily['avg_duration'] = df_daily['tripduration'] / df_daily['tripcount']
df_daily.head()


# ### Flags for Weekends and Holidays
# Add flags for weekends, bank holidays, and weekends that are within 2 days of holidays (a.k.a. holiday weekends).

# In[ ]:


# Create flags for weekends
df_hourly['weekend'] = np.where(df_hourly.day > 4 , 1, 0)
df_daily['weekend'] = np.where(df_daily.day > 4 , 1, 0)


# In[ ]:


# Ingest holidays 
bank_holidays = [
    '2014-01-01', '2014-01-20', '2014-02-17', '2014-05-26', 
    '2014-07-04', '2014-09-01', '2014-10-13', '2014-11-11', 
    '2014-11-27', '2014-12-25', '2015-01-01', '2015-01-19', 
    '2015-02-16', '2015-05-25', '2015-07-03', '2015-09-07', 
    '2015-10-12', '2015-11-11', '2015-11-26', '2015-12-25', 
    '2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', 
    '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', 
    '2016-11-24', '2016-12-25', '2017-01-02', '2017-01-16', 
    '2017-02-20', '2017-05-29', '2017-07-04', '2017-09-04', 
    '2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25'
]

# Add holiday flags
df_hourly['holiday'] = df_hourly['start_date'].apply(lambda x: 1*any([k in x for k in bank_holidays]))
df_daily['holiday'] = df_daily['start_date'].apply(lambda x: 1*any([k in x for k in bank_holidays]))

# Add holiday weekend flags (if weekend is within 2 days of a holiday)
cond = (df_daily['holiday'].shift(1) == 1) |          (df_daily['holiday'].shift(2) == 1) |          (df_daily['holiday'].shift(-1) == 1) |          (df_daily['holiday'].shift(-2) == 1) |          (df_daily['holiday'] == 1)

df_daily['weekend_holiday'] = np.where((cond * df_daily['weekend']) == 1, 1, 0)

# Add the second day of the holiday weekend if the other weekend day was flagged
cond = (df_daily['weekend_holiday'].shift(1) == 1) |          (df_daily['weekend_holiday'].shift(-1) ==  1) |          (df_daily['weekend_holiday'] == 1)
df_daily['weekend_holiday'] = np.where((cond * df_daily['weekend']) == 1, 1, 0)
df_daily.to_csv("holiday_weekends.csv")

# Finalize holiday weekend column
df_hweekends = df_daily[['start_date', 'weekend_holiday']]
df_hourly = pd.merge(df_hourly, df_hweekends, how = 'left', on = 'start_date')
df_hourly.head()


# In[ ]:


df_daily[['start_date','tripcount']].plot(kind = 'area', figsize = (18,5));


# In[ ]:


df_daily[['start_date','avg_duration']].plot(kind = 'area', figsize = (18,5));


# In[ ]:


# Add datetime columns for later use
df_hourly['start_datetime'] = pd.to_datetime(df_hourly['start_date'] + ' ' + df_hourly['hour'].astype(str) + ':00')
df_daily['start_datetime'] = pd.to_datetime(df_daily['start_date'])


# ## Exploratory Data Analysis

# ### Trip Counts

# In[ ]:


df_in.hist(figsize=(15, 18), bins=23, xlabelsize=10, ylabelsize=8);


# From this initial view, we can note a few observations and possible analyses that we can further dive into, for example:
# - **Growth Trend:** There seems to be an overall increase in the total volume of bike rentals, indicating there is likely a trend in the dataset
# - **Warm Temperature:** The temperature histogram shows that bike volume tend to increase when it's quite warm (between 60 and 80 degrees); however, as temperature increases beyond 80, the volume drops off since it's perhaps too hot out to be biking
# - **Monthly Seasonality:** The summer months seem to be the most popular time for bike rentals, which corroborates the finding above, which is that warmer temperatures correspond to higher volumes
# - **Hourly Seasonality:** The most popular times for bike rental occur around 9am and 5pm, which is rush hour for commuters
# - **Weekdays vs. Weekends:** Surprisingly, the weekday volumes look to be higher per day than the weekend volumes; this might be because a lot of bicycle renters are using them for commuting (more than the incremental volume that comes from leisure on the weekends)
# - **Location Concetration:** The vast majority of the rental activity is within a narrow range of longitude and latitude coordinates
# - **Short Rides:** The majority of the rides are fairly short (less than 15 minutes)

# In[ ]:


# Plot histograms for categorical columns 
fig, axs = plt.subplots(1,3, figsize=(15, 4))
axs = axs.ravel()
for i, categorical_col in enumerate(['usertype', 'gender', 'events']):
    df = df_in[categorical_col].value_counts() / df_in.shape[0]
    df.plot(kind='bar', title = categorical_col, ax = axs[i])


# From these charts we see that 1) the users of the service are predominately male, and 2) cloudy events see the largest amount of volume; however, this could be because the distribution of weather in chicago is skewed towards cloudy or that more rentals occur on cloudy days. This is a point we can test below, which shows the percentage of hours that the rides are disproportionately on cloudy days even though Chicago sees cloudy days almost 80% of the year. Rain or snow seems the most disproportionately low, followed by clear surprisingly. 

# In[ ]:


df_total = pd.DataFrame(df_hourly['events'].value_counts() / df_hourly.shape[0]).reset_index()
df_rides = pd.DataFrame(df_in['events'].value_counts() / df_in.shape[0]).reset_index()
df_total.columns = ['events', 'percentage_of_time']
df_rides.columns = ['events', 'percentage_of_rentals']
pd.merge(df_rides, df_total, on = 'events').set_index('events').plot(kind = 'bar');


# Next, we do a similar analysis for temperature. We see that the range between 70 to 80 degrees sees the most disproportionately high number rentals, with 80-90 next. The distribution begins to change at around 50 degrees where the rental distribution goes below the percentage of time distribution. 

# In[ ]:


tmp = pd.DataFrame(np.vstack((range(0,100,10), np.histogram(df_in['temperature'], range = (0,100))[0]/df_in.shape[0],
                              np.histogram(df_hourly['temperature'], range = (0, 100))[0]/df_hourly.shape[0])).T)
tmp.columns = ['degrees', 'percentage_of_rentals', 'percentage_of_time']
tmp = tmp.set_index('degrees')
tmp.plot(kind = 'bar');


# ### Trip Length

# In[ ]:


# Plot average for categorical columns 
fig, axs = plt.subplots(3,3, figsize=(15, 15))
axs = axs.ravel()
for i, categorical_col in enumerate(['year', 'month', 'week', 'day', 'temperature',
                                    'latitude_start', 'latitude_end', 'longitude_start', 'longitude_end']):
    b = len(df_in[categorical_col].value_counts())
    if b > 50 :
        tmp = df_in.groupby(pd.cut(df_in[categorical_col], min(b, 11)))['tripduration'].mean().reset_index()
        tmp[categorical_col] = [np.round((a.left + a.right)/2,1) for a in tmp[categorical_col]]
    else:
        tmp = df_in.groupby([categorical_col])['tripduration'].mean().reset_index()
    tmp = tmp.set_index(categorical_col)
    del tmp.index.name
    tmp.plot(kind = 'bar', title = categorical_col, ax = axs[i], legend = False, width = 0.9)


# From the average trip duration plot we can see a couple of interesting hyptheses worth testing:
# - **Warm weather:** higher temperature leads to longer trips, and does not generally taper off at the high temperatures as trip counts did
# - **Weekend trips:** weekend trips are longer than weekday trips on average, as they are likely to be for leisure instead of commute
# - **Seasonality:** as expected, the summer months has longer trips which also corresponds to when there are higher temperatures in Chicago
# 
# Another observation to note is that there is fairly little fluctuation in the average trip times whereas there is a lot more variation in the number of trips as seen previously. 

# In[ ]:


# Plot average for categorical columns 
fig, axs = plt.subplots(1,3, figsize=(15, 4))
axs = axs.ravel()
for i, categorical_col in enumerate(['usertype', 'gender', 'events']):
    tmp = df_in.groupby([categorical_col]).agg({'tripduration':np.mean})
    del tmp.index.name
    tmp.plot(kind='bar', ax = axs[i], title = categorical_col)


# Most weather events have pretty consistent average ride time of ~11min but does shortten significantly with rain or snow. 

# ### Feature Relationships
# 

# In[ ]:


scatter_matrix(df_daily[['month', 'day', 'temperature',
       'tripcount', 'avg_duration']], alpha=0.4, figsize=(18, 18), diagonal='kde', grid = True);


# This feature matrix corroborates some of the observations that we've made before around temperature being positively correlated with both trip count and duration. Looking at the bottom 3 graphs of the left-most column, we see that month is  correlated with temperature in a similar manner as trip count and duration. 
# 
# To test this, we run a correlation matrix on the 4 variables. Interestingly, month is shown to have fairly low correlation with the other 3 variables. However, temperature is confirmed to have very high correlation with trip count and duration.

# In[ ]:


tmp = df_hourly[['month', 'temperature', 'avg_duration', 'tripcount']].corr()
sns.heatmap(tmp, xticklabels=tmp.columns,yticklabels=tmp.columns, annot= True, linewidths = 0.5);


# ## Decomposing Trend and Seasonality
# 
# Here we use trend and seasonality decomposition to separate out the components of the trip count and duration time series. This is done for both the hourly and daily data. The residuals are then used for the regression analysis in the next section. 

# ### Trip Count
# Interestingly, the daily level decomposition for trip count looks more appropriate as it has more reasonable residuals; the hourly decomposition shows some cyclicality in the residuals. 

# In[ ]:


# Daily level
tc_counts = df_daily.set_index('start_datetime')[['tripcount']].tripcount
tc_daily = sm.tsa.seasonal_decompose(tc_counts, freq=365)
resplot = tc_daily.plot()


# In[ ]:


# Hourly level
tc_counts = df_hourly.set_index('start_datetime')[['tripcount']].tripcount
tc_hourly = sm.tsa.seasonal_decompose(tc_counts, freq=8760)
resplot = tc_hourly.plot()


# ### Trip Duration
# Both daily and hourly level decompositions look reasonable, which both show that in the summer months the average duration tends to be higher. There also seems to be somewhat of a trend that trip duration has been increasing, however, in the last year or so it has been flattening out or even decreasing a bit.

# In[ ]:


# Daily level
avg_dur = df_daily.set_index('start_datetime')[['avg_duration']].avg_duration
dur_daily = sm.tsa.seasonal_decompose(avg_dur, freq=365)
resplot = dur_daily.plot()


# In[ ]:


# Hourly level
avg_dur = df_hourly.set_index('start_datetime')[['avg_duration']].avg_duration
dur_hourly = sm.tsa.seasonal_decompose(avg_dur, freq=8760)
resplot = dur_hourly.plot()


# ## Regression Analysis
# 
# We now run linear regression analysis on both the residual data from the decomposition as well as the original data. Some dummy variables will need to be added for date and categorical fields. The coefficients and p-values for the weather variables are saved for summarization. 

# ### Data preparation

# In[ ]:


# Remove un-needed variables
df_hourly_regr = df_hourly.drop(['start_date', 'tripduration'], axis = 1).reset_index(level=0)
df_daily_regr = df_daily.drop(['start_date', 'tripduration'], axis = 1).reset_index(level=0)
df_hourly_regr.head()


# In[ ]:


# Create separate dataframes for residuals
df_daily_regr_resid = pd.merge(df_daily_regr, pd.DataFrame(dur_daily.resid).reset_index(), how = 'left', on = 'start_datetime')
df_daily_regr_resid = pd.merge(df_daily_regr_resid, pd.DataFrame(tc_daily.resid).reset_index(), how = 'left', on = 'start_datetime')

df_daily_regr_resid = df_daily_regr_resid[df_daily_regr_resid['avg_duration_y'].notnull()]
df_daily_regr_resid = df_daily_regr_resid.drop(['index', 'year', 'month', 'week', 'day', 'tripcount_x', 
                                                'avg_duration_x', 'weekend'], axis = 1)

df_hourly_regr_resid = pd.merge(df_hourly_regr, pd.DataFrame(dur_hourly.resid).reset_index(), how = 'left', on = 'start_datetime')
df_hourly_regr_resid = pd.merge(df_hourly_regr_resid, pd.DataFrame(tc_hourly.resid).reset_index(), how = 'left', on = 'start_datetime')

df_hourly_regr_resid = df_hourly_regr_resid[df_hourly_regr_resid['avg_duration_y'].notnull()]
df_hourly_regr_resid = df_hourly_regr_resid.drop(['index', 'year', 'month', 'week', 'day',
                                                  'hour', 'tripcount_x', 'avg_duration_x', 'weekend'], axis = 1)


df_hourly_regr_resid.head()


# In[ ]:


# Add dummy variables for time variables
for col in ['year', 'month', 'week', 'day']:
    df_hourly_regr[col] = df_hourly_regr[col].astype(str)
    df_daily_regr[col] = df_daily_regr[col].astype(str)
df_hourly_regr['hour'] = df_hourly_regr['hour'].astype(str)

df_daily_regr = pd.get_dummies(df_daily_regr, prefix = ['year', 'month', 'week', 'day', 'events'])
df_hourly_regr = pd.get_dummies(df_hourly_regr, prefix = ['year', 'month', 'week', 'day', 'hour', 'events'])

df_daily_regr_resid = pd.get_dummies(df_daily_regr_resid, prefix = ['events'])
df_hourly_regr_resid = pd.get_dummies(df_hourly_regr_resid, prefix = ['events'])


# Next we initialize the data structure that will hold the results so they'll be easily comparable.

# In[ ]:


# List of coefficients that we care about
weather_vars = ['temperature', 'events_clear', 'events_cloudy',
                'events_not clear', 'events_rain or snow', 'events_tstorms', 'events_unknown']

model_results = pd.DataFrame(columns = ['level', 'data_type', 'dependent_var', 'var', 'coef', 'pvalue'])

# Define function to append new results to dataframe
def add_to_results(m, level, d_type, d_var):
    out = model_results
    d = {}
    d['level'] = level
    d['data_type'] = d_type
    d['dependent_var'] = d_var
    for v in weather_vars:
        if v in model.params.index:
            d['var'] = v
            d['coef'] = m.params[v]
            d['pvalue'] = m.pvalues[v]
            out = out.append(d, ignore_index = True)
    return out


# ### Trip Count Analysis

# #### Regression with raw data and time dummies

# In[ ]:


# Daily Model - Raw data
X = df_daily_regr.drop(['avg_duration', 'tripcount', 'start_datetime'], axis = 1)
Y = df_daily_regr['tripcount']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'daily', 'original', 'tripcount')
model.summary()


# In[ ]:


# Hourly Model - Raw Data
X = df_hourly_regr.drop(['avg_duration', 'tripcount', 'start_datetime'], axis = 1)
Y = df_hourly_regr['tripcount']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'hourly', 'original', 'tripcount')
model.summary()


# #### Regression with decomposed residuals

# In[ ]:


# Daily Model - Residuals data
X = df_daily_regr_resid.drop(['avg_duration_y', 'tripcount_y', 'start_datetime'], axis = 1)
Y = df_daily_regr_resid['tripcount_y']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'daily', 'residuals', 'tripcount')
model.summary()


# In[ ]:


# Hourly Model - Raw Data
X = df_hourly_regr_resid.drop(['avg_duration_y', 'tripcount_y', 'start_datetime'], axis = 1)
Y = df_hourly_regr_resid['tripcount_y']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'hourly', 'residuals', 'tripcount')
model.summary()


# ### Trip Duration Analysis

# #### Regression with raw data and time dummies

# In[ ]:


# Daily Model - Raw Data
X = df_daily_regr.drop(['avg_duration', 'tripcount', 'start_datetime'], axis = 1)
Y = df_daily_regr['avg_duration']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'daily', 'original', 'avg_duration')
model.summary()


# In[ ]:


# Hourly Model - Raw Data
X = df_hourly_regr.drop(['avg_duration', 'tripcount', 'start_datetime'], axis = 1)
Y = df_hourly_regr['avg_duration']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'hourly', 'original', 'avg_duration')
model.summary()


# #### Regression with decomposed residuals

# In[ ]:


# Daily Model - Residuals data
X = df_daily_regr_resid.drop(['avg_duration_y', 'tripcount_y', 'start_datetime'], axis = 1)
Y = df_daily_regr_resid['avg_duration_y']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'daily', 'residuals', 'avg_duration')
model.summary()


# In[ ]:


# Hourly Model - Raw Data
X = df_hourly_regr_resid.drop(['avg_duration_y', 'tripcount_y', 'start_datetime'], axis = 1)
Y = df_hourly_regr_resid['avg_duration_y']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_results = add_to_results(model, 'hourly', 'residuals', 'avg_duration')
model.summary()


# The final summary of weather variable results are shown below, which show a general relationship between weather patterns both at the daily and hourly level on duration and number of trips. 

# In[ ]:


filtered_results = model_results[model_results['pvalue']<0.05]
pivot_results = pd.pivot_table(filtered_results, index = ['dependent_var','var'], columns = ['level', 'data_type'], values = 'coef', aggfunc = np.mean, fill_value = 'n/a')
pivot_results

