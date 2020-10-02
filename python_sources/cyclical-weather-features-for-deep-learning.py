#!/usr/bin/env python
# coding: utf-8

# # Encoding Cyclical Features for Deep Learning

# In this notebook we are going to look at how to properly encode cyclical features for use in deep learning.
# 
# Many features commonly found in datasets are cyclical in nature. The best example of such a feature is of course *time*: months, days, weekdays, hours, minutes, seconds etc. are all cyclical in nature. Other examples include features such as seasons, tidal and astrological data. 
# 
# The problem is letting the deep learning algorithm know that features such as these occur in cycles.
# 
# We will be exploring this problem at the hand of an example: predicting temperature from historical weather data for the city of Montreal.
# 
# If you would like to skip the data gathering and pre-processing part, feel free to go straight to the [cyclical feature](#Cyclical-Features) section. A more concise version of the general idea is given at [https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/](https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/).

# ## Preparing the Data

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')


# The dataset is divided into separate files, one per weather attribute. To illustrate the encoding of cyclical features we will only be using the temperature data of a single city: Montreal.

# In[2]:


cities_temperature = pd.read_csv("../input/temperature.csv", parse_dates=['datetime'])
cities_temperature.sample(5)


# Each city's temperature time series is given as a column labeled by the city name. It also seems like the temperature is given in Kelvin. Let's get Montreal's temperature data, along with the corresponding datetime. We double check that pandas imported datetime as a datetime type and also convert the temperature to Celsius to make it more relatable.

# In[3]:


data = cities_temperature[['datetime', 'Montreal']]
data = data.rename(columns={'Montreal': 'temperature'})
data['temperature'] = data['temperature'] - 273.15
print(data.dtypes)


# In[4]:


data.head(5)


# Great, we see that there are some missing values. Temperature tends to be relatively constant from one hour to the next, so we can use backfill to replace any missing values and simply drop the rest.

# In[5]:


data = data.fillna(method = 'bfill', axis=0).dropna()


# In[6]:


print(data.temperature.describe())
ax = sns.distplot(data.temperature)


# The temperature data looks good. Let's have a look at the cyclical features.

# ## Cyclical Features

# We can now illustrate the problem of time as a cyclical feature. We will do so using the hour of the day as an example. Let's extract the hours from the datetime:

# In[7]:


data['hour'] = data.datetime.dt.hour
sample = data[:168] # roughly the first week of the data


# In[8]:


ax = sample['hour'].plot()


# Here we see exactly what we would expect from hourly data for a week: a cycle between 0 and 23 that repeats 7 times.
# 
# This graph also illustrates the problem with presenting cyclical data to a machine learning algorithm: there are *jump discontinuities* in the graph at the end of each day, when the hour value goes from $23$ to $00$.
# 
# Let's have a look at the time period around midnight of the first day:

# In[9]:


sample[9:14]


# The difference in time between records 10 and 11 is of course $1$ hour. If we leave the `hour` feature unencoded, everything works in this case: $23-22=1$
# 
# However, if we look at rows 11 and 12 we see the failure in our encoding: $0 - 23= -23$, even though the records are again only one hour apart.
# 
# *We need to change the encoding of the feature such that midnight and 11:00PM are the same distance apart as any other two hours.*

# ### Encoding Cyclical Features

# A common method for encoding cyclical data is to transform the data into two dimensions using a sine and consine transformation.
# 
# We can do that using the following transformations:
# 
# $x_{sin} = \sin(\frac{2 * \pi * x}{\max(x)})$
# 
# $x_{cos} = \cos(\frac{2 * \pi * x}{\max(x)})$
# 
# Let's do this for our hourly data:

# In[10]:


data['hour_sin'] = np.sin(2 * np.pi * data['hour']/23.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour']/23.0)


# Why  two dimensions, using sine and cosine you may ask? Let's have a look at just one dimension:

# In[11]:


sample = data[0:168]


# In[12]:


ax = sample['hour_sin'].plot()


# As expected, it is cyclical, based on the sine graph. Looking at the values around midnight again:

# In[13]:


sample[10:26]


# Great, it appears the absolute difference an `hour_sin` before, at and after midnight is now the same! However, if we look at the plot of `hour_sin` (following any flat line intersection with the graph), we can see there is a problem. If we consider just the one dimension, there are two records with exactly the same `hour_sin` values, e.g. records 11 and 25.
# 
# This is why we also need the cosine transformation, to separate these records from each other.
# 
# Indeed, if we plot both features together in two dimensions we get the following:

# In[14]:


ax = sample.plot.scatter('hour_sin', 'hour_cos').set_aspect('equal')


# Exactly what we want: our cyclical data is encoded as a cycle. Perfect for presenting to our deep learning algorithm.

# ## Learning from Encoded Data

# Let's see how the encoding affects a deep neural network. We will create a model that attempts to predict the temperature in Montreal for a given month, day and hour.
# 
# We need to add the extra cyclical features and also encode them using our sine and cosine transformation. We'll create a small helper function for that.

# In[15]:


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


# In[16]:


data['month'] = data.datetime.dt.month
data = encode(data, 'month', 12)

data['day'] = data.datetime.dt.month
data = encode(data, 'day', 365)


# In[18]:


data.head()


# In[ ]:


data.to_csv("montreal_hourlyWeather_cyclicEncoded.csv.gz",index=False,compression="gzip")


# Let's split our data into training and test sets. Since we will be comparing models, we also create a validation set.

# In[ ]:


from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(data, test_size=0.4)
data_test, data_val = train_test_split(data_test, test_size=0.5)


# #### Building the model

# The model we will be using will be a simple three layer deep neural network with rectified linear unit activation functions in the hidden layer and a linear activation function in the output layer. We will use an [Adam](https://arxiv.org/abs/1412.6980) optimizer with the default learning rate of $0.1$. We will use the mean squared error as loss function.

# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam

def train_model(X_train, y_train, X_test, y_test, epochs):
    model = Sequential(
        [
            Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
#             Dense(10, activation="relu"),
#             Dense(10, activation="relu"),
            Dense(1, activation="linear")
        ]
    )
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model, history


# ### Unencoded features

# First, we will build a model using the unencoded features.

# In[ ]:


def get_unencoded_features(df):
    return df[['month', 'day', 'hour']]


# In[ ]:


X_train = get_unencoded_features(data_train)
X_test  = get_unencoded_features(data_test)
y_train = data_train.temperature
y_test  = data_test.temperature


# In[ ]:


model_unencoded, unencoded_hist = train_model(
    get_unencoded_features(data_train),
    data_train.temperature,
    get_unencoded_features(data_test),
    data_test.temperature,
    epochs=5
)


# ### Encoded features

# Next, we train the same architecture using the encoded features.

# In[ ]:


def get_encoded_features(df):
    return df[['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']]


# In[ ]:


X_train = get_encoded_features(data_train)
X_test  = get_encoded_features(data_test)
y_train = data_train.temperature
y_test  = data_test.temperature


# In[ ]:


model_encoded, encoded_hist = train_model(
    get_encoded_features(data_train),
    data_train.temperature,
    get_encoded_features(data_test),
    data_test.temperature,
    epochs=5
)


# #### Comparison

# Immediately evident is a dramatic improvement in our model's convergence rate. After only one epoch, the model using the encoded features has a validation loss comparable to the unencoded model's final validation loss.

# In[ ]:


plt.plot(unencoded_hist.history['val_loss'], "r")
ax = plt.plot(encoded_hist.history['val_loss'], "b")


# How do the models compare on the validation data?

# In[ ]:


X_val_unencoded  = get_unencoded_features(data_val)
X_val_encoded  = get_encoded_features(data_val)
y_val = data_val.temperature


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mse_unencoded = mean_squared_error(y_val, model_unencoded.predict(X_val_unencoded))
print(mse_unencoded)


# In[ ]:


mse_encoded = mean_squared_error(y_val, model_encoded.predict(X_val_encoded))
print(mse_encoded)


# In[ ]:


print('We achieved an improvement of {0:.2f}% in our MSE'.format((mse_unencoded - mse_encoded)/mse_unencoded * 100))


# ## Summary

# It's important to encode features correctly for the specific machine learning algorithm being used. Other machine learning algorithms might be robust towards raw cyclical features, particularly tree-based approaches. However, deep neural networks stand to benefit from the encoding strategy discussed above, particularly in terms of the convergence speed of the network.

# In[ ]:




