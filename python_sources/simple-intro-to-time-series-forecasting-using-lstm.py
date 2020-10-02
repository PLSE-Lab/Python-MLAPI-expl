#!/usr/bin/env python
# coding: utf-8

# **Import Libraries**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# **Read the dataset**

# In[ ]:


df = pd.read_csv('../input/daily_total_female_births_in_cal.csv')
df.columns = ['Births']
df_copy = df.copy()
df.head()


# Now lets visualize the whole dataset

# In[ ]:


sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(df['Births'])


# **Convert Time-Series Problem into a Supervised Learning Problem**
# 
# To do that, we need input variable(s) and an output variable. We can transform the data and use previous time steps as input variables and use the next time time step as the output variable. 
# 
# For example: If the dataset looked like
# 
# 1
# 
# 2
# 
# 3 
# 
# We can restructe it into a supervised learning problem by writing it in the following format
# 
# 1, 2
# 
# 2, 3
# 
# where the first column is the input and the second column is the output.
# 
# To do the above programmatically, we can use the pandas shift function which basically pulls or pushes a column by a set number of rows.

# In[ ]:


df['Births(t-0)'] = df.Births.shift(-1)
df.head()


# As we can see above, using -1 pulls the column up by one row. Here '-' is for pulling the column up and 1 instructs pandas by how many columns it should be pulled up.

# In[ ]:


df['Births(t-1)'] = df.Births.shift(-2)
df.head()


# We did the same thing again but this time with -2 which generated another column
# The last column is the target or output variable and before that we can use as many time steps as we wish depending on the number of time-steps that makes a better model.

# In[ ]:


df['Births(t-2)'] = df.Births.shift(-3)
df['Births(t-3)'] = df.Births.shift(-4)
df.head()


# For this problem, we will use 4 previous timesteps to predict the 5th timestep
# 
# But since the columns are being pulled from the top, certain cells at the bottom will be empty so we will remove any rows that have NaN values

# In[ ]:


df = df.dropna()


# Now, we divide the data into training and testing set. 
# Last 30 rows for testing and the rest for training. 
# 
# The last column is the output variable and the rest are the input variables

# In[ ]:


test_set = df[-30:]
train_set = df[:-30]
X_test_set = test_set.drop(['Births(t-3)'], axis=1)
y_test_set = test_set['Births(t-3)']
X_train_set = train_set.drop(['Births(t-3)'], axis=1)
y_train_set = train_set['Births(t-3)']


# Again import libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[ ]:


X = array(X_train_set)
y = array(y_train_set)


# We have to reshape X into [samples, timesteps, features]
# 
# samples -> no of rows there are
# 
# timesteps -> the number of timesteps we are using for prediction
# 
# features -> here features will be 1 because this is a univariate forecasting problem and we are using only one feature i.e. female births

# In[ ]:


X = X.reshape((X.shape[0], X.shape[1],1))


# Define a LSTM model and fit the data

# In[ ]:


model = Sequential()
model.add(LSTM(128, activation = 'relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, validation_split=0.10)


# Predict the output for the test set

# In[ ]:


x_test = array(X_test_set)
x_input = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
prediction = model.predict(x_input, verbose=0)
print(prediction)


# Convert the actual outputs and the predicted outputs into 1D array

# In[ ]:


y_test_set = array(y_test_set)
y_test_set


# In[ ]:


prediction.flatten()


# Lets visualize the actual output and the predicted output

# In[ ]:


sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(y_test_set, label = "Actual")
plt.plot(prediction, label = "Predicted")
plt.show()


# Lets visualize how the whole time series looks with  the predicted ouputs

# In[ ]:


actual = np.append(y,y_test_set)
predicted = np.append(y, prediction)


# In[ ]:


sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(actual, label = "Actual")
plt.plot(predicted, label = "Predicted", alpha=0.7)
plt.legend()
plt.show()


# The last 30 data points are where we tested our model and it did predict the ups and downs to some extent. 
# 
# The model can be improved by playing with the neural net architecture and the number of timesteps that are used for predicting or with more data
# 
# **Please leave an upvote :D**
