#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing data and cleaning

# ## Importing data

# Data is imported as .csv from Kaggle and imported with pandas library

# In[ ]:


data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv', low_memory = False)


# In[ ]:


data.head(5)


# ## Exploring data

# ### Number of unique members belonging to different dimensions

# In[ ]:


print('No. of Regions: %i' %data['Region'].nunique())
print('No. of Countries: %i' %data['Country'].nunique())
print('No. of States: %i' %data['State'].nunique())
print('No. of Cities: %i' %data['City'].nunique())


# ### Members of various dimensions

# In[ ]:


print(data["Region"].unique())
print(data["Country"].unique())
print(data["Month"].unique())
print(data["Day"].unique())
print(data["Year"].unique())


# ### Displaying basic information about data

# In[ ]:


data.info()


# ### Descriptive statistics of measures

# Here we find that the minimum Average temperaure temperature is -99 degrees which is unusual

# In[ ]:


data.describe()


# ## Counting N/As in data

# Counting the number of N/As, we find that the states is only defined for the U.S.A. Hence we need to drop the column.

# In[ ]:


data.isna().sum()


# # Cleaning Data

# ## Removing data

# The following procedures are performed to clean the data
# 
# * The rows with values '201' and '200' are removed from the Years column.
# * The rows with day as '0' is removed.
# * The States column with data only for the US is removed.
# * The rows with temperature -99 degress Fahrenheit is removed 

# In[ ]:


# Removing '201' and '200' from Year column
df = data[~data['Year'].isin(['201','200'])]

# Removing '0' from Date column
df = df[df['Day'] != 0]

# Removing 'State' column
df = df.drop(columns=['State'])

# Dropping the rowns with temperature -99
df = df.drop(df[df['AvgTemperature'] == -99.0].index)


# ## Adding data/columns

# The following columns are added
# 
# * Average Temperature in Celcius is added.
# * Combining year, month and day to make the Date column in the format YYYY-MM-DD
# * Introduce column Period in the format YYYY-MM

# In[ ]:


# Adding a row with average temperatures in Celcius
df['AvgTempCelcius'] = round((((df.iloc[:,6] - 32) * 5) / 9),2)

# Adding the Date column in the format YYYY-MM-DD
df['Date'] = df.iloc[:,5].astype(str) + '-' + df.iloc[:,3].astype(str) + '-' + df.iloc[:,4].astype(str)

# Coverting the Date column into Pandas Date type datetime64[ns]
df['Date'] = pd.to_datetime(df['Date'])

# Introducing the Period column in format YYYY-MM
df['Month/Year'] = pd.to_datetime(df['Date']).dt.to_period('M')


# In[ ]:


df.info()


# In[ ]:


df.head()


# # Visualization and Exploratory Data Analysis

# ## Mean Average temperature of different continents in Fahrenheit

# In[ ]:


print(df.groupby(['Region'])['AvgTemperature'].mean())
avg_temp_world = pd.Series(round(df.groupby('Region')['AvgTemperature'].mean().sort_values(),2))
avg_temp_world.plot(kind='bar', figsize = (10,6), color='yellow', alpha=0.5)
plt.xlabel('Mean Average Temperature')
plt.ylabel('Regions')
plt.title('Mean Average Temperatures by Region')


# ### Mean Average temperature in the world from 1995 to 2019

# In[ ]:


world_temp_date = pd.DataFrame(pd.Series(round(df.groupby('Date')['AvgTempCelcius'].mean(),2))[:-1])
world_temp_year = pd.DataFrame(pd.Series(round(df.groupby('Year')['AvgTempCelcius'].mean(),2))[:-1])

plt.subplot(2,1,1)
sns.set_style("darkgrid")
sns.lineplot(data = world_temp_date, color = 'blue')
plt.xlabel('Time')
plt.ylabel('Temperature (in Celcius)')
plt.title('Mean Avg. Temperature Over Time (Date) in the world')
plt.show()

plt.subplot(2,1,2)
sns.set_style("darkgrid")
sns.lineplot(data = world_temp_year, color = 'blue')
plt.xlabel('Time')
plt.ylabel('Temperature (in Celcius)')
plt.title('Mean Avg. Temperature Over Time (Years) in the world')
plt.show()


# ### Average Temperatures for various countries over time from 1995 to 2019

# In[ ]:


## Creating a function to plot the temperature over the Periods in different countries
def plot_temp_country_month(country, format = '-', temp = 'Celcius'):
    dat = df[df['Country'] == country]
    if temp == 'Celcius':
        dat_temp = pd.Series(round(dat.groupby('Date')['AvgTempCelcius'].mean().sort_values(),2))
    else:
        dat_temp = pd.Series(round(dat.groupby('Date')['AvgTemperature'].mean().sort_values(),2))
    sns.set_style("darkgrid")
    sns.lineplot(data = dat_temp, color = 'red')
    plt.xlabel('Time (Periods)')
    plt.ylabel('Temperature (in %s)' %temp)
    plt.title('Mean Avg. Temperature Over Time in %s' %country)
    plt.show()

    
## Creating a function to plot the temperature over the Years in different countries
def plot_temp_country_year(country, format = '-', temp = 'Celcius'):
    dat = df[df['Country'] == country]
    if temp == 'Celcius':
        dat_temp = pd.DataFrame(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTempCelcius'].mean(),2))[:-1])
    else:
        dat_temp = pd.DataFrame(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTemperature'].mean(),2))[:-1])
    sns.set_style("darkgrid")
    sns.lineplot(data = dat_temp, color = 'red' , style = 'event', hue = 'cue')
    plt.xlabel('Time (Years)')
    plt.ylabel('Temperature (in %s)' %temp)
    plt.title('Mean Avg. Temperature Over Time in %s' %country)
    plt.show()


# In[ ]:


plot_temp_country_year('India')
plot_temp_country_month('India')


# ## Temperature Fluctuation over time from 1995 to 2019

# In the section the temperature fluctuations from 1996 to 2019 relative to the year 1995 are calculated and visualised. Below are the functions defined:
# 
# Note: The analysis has been carried out in Celcius scale 

# In[ ]:


## Function to calculate the temperature fluctuation
def calculate_fluctuation(series):
    fluctuation = np.zeros((len(series),))
    for i in range(1,len(series)):
        fluctuation[i] = series[i] - series[0]
    return fluctuation

## Function to plot the temperature fluctuation Lineplot
def plot_change(years, fluctuation, entity):
    change_df = pd.DataFrame(np.column_stack((years, fluctuation)), columns = ['Year', 'Change'])
    change_df['Year'] = change_df['Year'].astype(int)
    sns.lineplot(x = "Year", y = "Change", data = change_df, err_style="bars", ci=68, label = entity)
    x = np.zeros((len(change_df['Year']),1))
    plt.plot(change_df['Year'], x, '--')
    plt.title('Temperature fluctuations over the years')
    plt.ylabel('Change in Temperature')

## Function to plot the temperature fluctuation in the world 
def world_change_temp():
    dat = np.array(pd.Series(round(df.groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]
    years = np.arange(1995,2020)
    fluctuation = calculate_fluctuation(dat)
    plot_change(years, fluctuation, 'World')

## Function to plot the temperature fluctuation in every country
def country_change_temp(country):
    dat = np.array(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]
    years = np.arange(1995,2020)
    fluctuation = calculate_fluctuation(dat)
    plot_change(years, fluctuation, country)

## Function to plot the temperature fluctuation in every continent
def region_change_temp(region):
    dat = np.array(pd.Series(round(df[df['Region'] == region].groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]
    years = np.arange(1995,2020)
    fluctuation = calculate_fluctuation(dat)
    plot_change(years, fluctuation, region)

## Function to plot the average temperature and temperature fluctutaion distribution for every country
def temperature_histogram(country):
    hist1_s = np.array(pd.Series(round(df.groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]
    hist2_s = np.array(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]
    hist1_fluctuation = calculate_fluctuation(hist1_s)
    hist2_fluctuation = calculate_fluctuation(hist2_s)
    print('Skewness for Temperature in %s: ' %country, df[df['Country'] == country]['AvgTempCelcius'].skew())
    print('Kurtosis for Temperature in %s: ' %country, df[df['Country'] == country]['AvgTempCelcius'].kurt())
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    sns.distplot(df[df['Country'] == country]['AvgTempCelcius'], label = country)
    sns.distplot(df['AvgTempCelcius'], label = 'World')
    plt.legend()
    plt.subplot(1,2,2)
    sns.distplot(hist1_fluctuation , label = 'World')
    sns.distplot(hist2_fluctuation, label = country)
    plt.xlabel('Temperature Fluctuations')
    plt.legend()


# ## Temperature fluctuation in the world

# The Average world temperature has continually increased after 1995. There have been ups and downs. One can also find that the temperature has dropped during the 2008 economic crisis. 

# In[ ]:


world_change_temp()


# ## Temperature fluctuation in a country

# From the below graphs, we can find that there is a gradual increase in the temprerature from 1995 in the United States and China, but temperature has dropped in Canada. Also, we can find that the temperature dropped during the 2008 economic crisis. 

# In[ ]:


country_change_temp('US')


# In[ ]:


country_change_temp('Canada')


# In[ ]:


country_change_temp('China')


# ## Temperature fluctuation in a continent

# In[ ]:


region_change_temp('Africa')


# ## Temperature and fluctuation distribution in a country

# In[ ]:


temperature_histogram('US')


# # Time - Series Forecasting Model with DNN, LSTM and CNN

# In order to create a forecasting model, the data has been considered as a time-series data and Neural networks are used to recognise and predict time series

# In[ ]:


# Create the series
series = np.array(list(pd.Series(round(df.groupby('Date')['AvgTempCelcius'].mean(),2))))

# Creating time intervals
time = np.array(np.arange(0,len(series)))


# In[ ]:


# We have 7000 training examples and rest as training examples
split_time = 7000

# Defining the Training set and test set
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Initialising the Hyperparamenters
window_size = 60
batch_size = 100
shuffle_buffer_size = 1000


# ## Defining functions to feed into the training Model

# In[ ]:


## Function to create a line plot
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

## Function to prepare data to be fed into the Tensorflow model
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dp = tf.data.Dataset.from_tensor_slices(series)
    dp = dp.window(window_size + 1, shift=1, drop_remainder=True)
    dp = dp.flat_map(lambda w: w.batch(window_size + 1))
    dp = dp.shuffle(shuffle_buffer)
    dp = dp.map(lambda w: (w[:-1], w[1:]))
    return dp.batch(batch_size).prefetch(1)

## Function to prepare validation data into the model for prediction
def model_forecast(model, series, window_size):
    dp = tf.data.Dataset.from_tensor_slices(series)
    dp = dp.window(window_size, shift=1, drop_remainder=True)
    dp = dp.flat_map(lambda w: w.batch(window_size))
    dp = dp.batch(32).prefetch(1)
    forecast = model.predict(dp)
    return forecast


# ## Plotting Time series

# In[ ]:


# Plotting the time series
plot_series(time, series)


# ## Defining and running the model with callbacks to tune Learning rate for SGD optimizer

# In[ ]:


## The model contains 1 ConV1D filter, 2 LSTMs, 3 Dense layers and 1 Lambda layer

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

## Defining window sizes for hyperparameter tuning 
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)

# Defining the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

# Create a callback function to get optimal learning rate
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

# Defining the optimizer
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

# Compiling the model with Huber loss function 
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Running the model
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])


# ## Plotting learning rate vs loss

# Plotting a line plot of Learning rate vs loss to find the learning rate with minimal loss. Here we can see that at e-6 the loss is very low. Hence the learning rate is chosen to be 1e-6

# In[ ]:


plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 60])


# ## Building the model and training

# In[ ]:


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)

# Defining the model with 1 ConV1D filter, 2 LSTMs, 3 Dense layers and 1 Lambda layer
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

# Defining the optimizer with learning rate 1e-6 and momentum 0.9
optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)

# Compiling model with Huber loss function
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Training with 50 epochs
history = model.fit(train_set,epochs=50)


# ## Creating a forecast for the validation data

# In[ ]:


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# ## Plotting the validation data and the forecast

# In[ ]:


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)


# ## Checking the mean absolute error of the validation set

# We can find that the MAE for the validation set is pretty low and hence the model is a good forcasting model

# In[ ]:


tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

