#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

sns.set()


# ## Read data

# In[ ]:


df_covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# # Exploring data and preprocessing

# ## Force columns to be snake case

# In[ ]:


cols = df_covid.columns
df_covid.columns = [col.lower() for col in cols]


# In[ ]:


df_covid.rename(columns={
    'observationdate' : 'observation_date',
    'country/region' : 'country',
    'province/state' : 'province_state', 
    'last update' : 'last_update',
}, inplace=True)


# ## Transform date column to date format
# 
# Even that the dataset looks sorted by date, there's no harm in sorting it again

# In[ ]:


df_covid['observation_date'] = pd.to_datetime(df_covid['observation_date'])

df_covid.sort_values('observation_date', inplace=True)


# ## Group by day
# 
# This notebook intend to prepare data for worldwide daily forecast. Therefore, we're going to group by day without worring much of different countries and number of days since fist occurence. 

# In[ ]:


df_covid['diseased'] = df_covid['confirmed'] - df_covid['recovered'] - df_covid['deaths']

df_series = df_covid.groupby('observation_date').agg({
    'country' : 'nunique',
    'confirmed' : 'sum',
    'deaths' : 'sum',
    'recovered' : 'sum',
    'diseased' : 'sum',
})


# ## Drop unwanted columns

# In[ ]:


df_covid.drop(['sno', 'last_update'], axis=1, inplace=True)


# ## Create lags

# In[ ]:


for i in range(7, 15):
    df_series[f'confirmed_lag_{i}'] = df_series['confirmed'].shift(i)
    df_series[f'deaths_lag_{i}'] = df_series['deaths'].shift(i)
    df_series[f'recovered_lag_{i}'] = df_series['recovered'].shift(i)
    df_series[f'diseased_lag_{i}'] = df_series['diseased'].shift(i)


# ## Diagonal correlation matrix

# In[ ]:


sns.set(style="white")

fig, ax = plt.subplots(figsize=(11, 9))

# Create correlation matrix
corr = df_series.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

# Plot correlation matrix without the diagonal and upper part
sns.heatmap(corr, mask=mask, cmap=cmap, linewidths=.5)


# As expected, we can see that there is correlation between number of deaths and cases confirmed, among other corelations. It looks like our country information it's a little bit naive since countries have different density of people and also different months since first covid-19 occurence.

# ## Create percentage columns
# 
# I do always like to see percentage values, absolute values sometimes is hard to infere anything because you don't have a scale in your head

# In[ ]:


sns.set()

# Get percentages of recovered, deaths and diseased
df_series['pct_recovered'] = round(df_series['recovered'] / df_series['confirmed'], 4)
df_series['pct_deaths'] = round(df_series['deaths'] / df_series['confirmed'], 4)
df_series['pct_diseased'] = round(df_series['diseased'] / df_series['confirmed'], 4)


# ## Create percentage change columns
# 
# Also, I do like to see how was the percentage change between the observation and the value in the last observation. Don't use this column in your forecast, that would be cheating!

# In[ ]:


df_series['country_pct_change'] = df_series['country'].pct_change()
df_series['recovered_pct_change'] = df_series['recovered'].pct_change()
df_series['deaths_pct_change'] = df_series['deaths'].pct_change()
df_series['diseased_pct_change'] = df_series['diseased'].pct_change()


# ### Series for top 10 countries

# In[ ]:


df_country = df_covid.groupby(['country', 'observation_date']).sum().reset_index()

df_country['confirmed_log'] = np.log10(df_country['confirmed'])
df_country['deaths_log'] = np.log10(df_country['deaths'])


# #### Get column with number of days since the first occurency

# In[ ]:


df_country['day_cnt'] = 0 
for country in df_country['country'].unique():
    day_cnt = [i for i in range(1, df_country[df_country['country'] == country][df_country['confirmed'] > 0].shape[0] + 1)]
    
    df_country.loc[(df_country['country'] == country) & (df_country['confirmed'] > 0) , 'day_cnt'] = day_cnt


# #### Confirmed cases

# In[ ]:


# Column to plot
target = 'confirmed_log'

# It'd be better if this dataframe was outside this cell, but that's fine
countries_to_plot = df_country.groupby('country').sum().sort_values(target, ascending=False).index[:10]

# Plot top 10
plt.figure(figsize=(13, 11))
sns.lineplot(x='day_cnt', y=target, hue='country', data=df_country[df_country[target] > 0][df_country['country'].isin(countries_to_plot)])

plt.show()


# ## Deaths

# In[ ]:


# Column to plot
target = 'deaths'

# It'd be better if this dataframe was outside this cell, but that's fine
countries_to_plot = df_country.groupby('country').sum().sort_values(target, ascending=False).index[:10]

# Plot top 10
plt.figure(figsize=(13, 11))
sns.lineplot(x='day_cnt', y=target, hue='country', data=df_country[df_country[target] > 0][df_country['country'].isin(countries_to_plot)])

plt.show()


# ## Deaths percentage change with countries 
# 
# I wanted to see two things with the following plot
# 
# 1. Growth of number of deaths; 
# 2. How the number of countries is affecting my growth of deaths;
# 
# After adding the country_pct_change i've looked that in the first few rows my number of countries deacresed, making no sense, this is probably an error of collected data due that it should be cummulative

# In[ ]:


df_series[['deaths_pct_change', 'country_pct_change']].plot(figsize=(14, 5))


# # Time series forecasting
# 
# Let's forecast the number of deaths for the next 7 days for worlwide just for fun.

# ## Divide in train and test

# In[ ]:


df_series.dropna().shape


# In[ ]:


train_cols = [col for col in df_series.columns if 'deaths_lag_' in col] 

# Just to see if i'm not cheating ;D
', '.join(train_cols) 


# In[ ]:


num_split = 7

X = np.log10(df_series.dropna()[train_cols])
y = np.log10(df_series.dropna()['deaths'])

X_train = X[:-num_split]
y_train = y[:-num_split]
X_test = X[-num_split:]
y_test = y[-num_split:]


# ## Forecast
# 
# We're going to use multiple linear regression to predict the number of deaths for the next week.

# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)


# ## Evaluate model

# In[ ]:


predictions = model.predict(X_test)

df_predictions = pd.DataFrame()
df_predictions['y_pred_log'] = predictions
df_predictions['y_true_log'] = y_test.values
df_predictions['y_pred'] = 10 ** predictions
df_predictions['y_true'] = 10 ** y_test.values

df_predictions['absolute_pct_error'] = abs((df_predictions['y_pred'] - df_predictions['y_true']) / df_predictions['y_true']) * 100


# ### MAPE
# 
# Mean absolute percentage error

# In[ ]:


f"MAPE: {round(df_predictions['absolute_pct_error'].mean())}%"


# Well, I'd not recommend it for any application in real life! Let's see how it did individually

# ### Plot prediction vs true value

# #### Log values

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(y_train, 'bo--')
ax.plot(y_test, 'go--')
ax.plot(pd.Series(predictions, index = y_test.index), 'ro--')

plt.title('Log10 values')
plt.show()


# #### Absolute values

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(10 ** y_train, 'bo--')
ax.plot(10 ** y_test, 'go--')
ax.plot(10 ** pd.Series(predictions, index = y_test.index), 'ro--')

plt.title('Absolute values')
plt.show()


# ### Predictions and error individually

# In[ ]:


df_predictions['y_pred'] = round(df_predictions['y_pred'])
df_predictions


# Prediction might be better if we try to do one day validation for 7 days or use more features (after doing tsa). Also, it's always good to remember that this is just a biased validation for demonstration purposes. In cases that you really want to validate your forecast, I'd recommend implementing a time series cross-validation (not same as regular cross val)
