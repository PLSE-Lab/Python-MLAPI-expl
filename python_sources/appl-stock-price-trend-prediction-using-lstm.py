#!/usr/bin/env python
# coding: utf-8

# ### Hello guys, today we will see how to predict the open stock price of APPL Stock. Wait a sec. What !! "predict the open stock price". Is that even possible?            Ofcourse not. Its impossible to predict open stock price otherwise we would all become billionaire.                                                                                                                                                                             
# ### In this tutorial we will try to predict the upward and downward trends that exists in the APPL Stock Price, i.e. we are more interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price.                                                                                                                                                                                               
# 
# ###                                                                                                              We will not use time series forecasting, rather use Long short-term memory (LSTM) for our prediction. It performs well better than than the ARIMA model. We will design a stacked LSTM having high dimensionality.                                                                                                                         
# 
# ###    We will train our LSTM model using 6 years of APPL stock price , starting from the last financial date of 2009 to last financial day of 2016. Based on the correlations captured by our model, we will try to predict the open stock price from 1st financial day of 2017 to last financial day of August 2018. Again, we won't predict the actual price but try to predict the upward and downward stock price trends.
# ### So, without wasting more time, let's get started.

# ## Importing the libraries.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import math
from sklearn.metrics import mean_squared_error


# ## Loading APPL Stock Price train and test dataset

# In[ ]:


train_data = pd.read_csv("../input/train-data/train.csv", header=0)


# In[ ]:


train_data.head()


# In[ ]:


test_data = pd.read_csv("../input/test-data/test.csv")


# In[ ]:


test_data.head()


# ## Visualizing train data

# In[ ]:


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data_train = pd.read_csv('../input/train-data/train.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)


# In[ ]:


ts = data_train['Open'] 
plt.xlabel('Dates')
plt.ylabel('Open Prices')
plt.plot(ts)


# ## Visualizing test data Open Price trends with respect to date that will be predicted by our RNN model

# In[ ]:


data_test = pd.read_csv('../input/test-data/test.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)


# In[ ]:


ts = data_test['Open'] 
plt.xlabel('Dates')
plt.ylabel('Open Prices')
plt.plot(ts)


# In[ ]:


train = train_data.iloc[:, 1:2].values


# ## Feature Scaling

# In[ ]:


scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)


# ## Now we will creating a data structure with 60 timesteps and 1 output, i.e.Open Stock Price 

# In[ ]:


X_train = []
y_train = []
for i in range(60, train.shape[0]):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# ## Reshaping
# *  ### Here second argument is (batch_size, time_step ,input_dim)
# * ### batch_size is total number of stock price from 2009-12-31 to 2016, i.e. given by X_train.shape[0]
# * ### time_step is total number of previous stock price we want to consider while predicting present stock price, i.e given by X_train.shape[1]
# * ### third argument is input_dim-in our case it is 1, i.e.Open price, but it can be more than one. It basically includes all those factors/indicators that can affect present stock price 

# In[ ]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# ## Building the RNN

# In[ ]:


model = Sequential()

# Adding the first LSTM layer 
# Here return_sequences=True means whether to return the last output in the output sequence, or the full sequence.
# it basically tells us that there is another(or more) LSTM layer ahead in the network.
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout regularisation for tackling overfitting
model.add(Dropout(0.20))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.25))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.25))

model.add(LSTM(units = 50))
model.add(Dropout(0.25))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
# RMSprop is a recommended optimizer as per keras documentation
# check out https://keras.io/optimizers/ for more details
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)


# ## Now making the predictions and visualising the results

# In[ ]:


# this will be used later while comparing and visualization
real_stock_price = test_data.iloc[:,1:2].values


# In[ ]:


# combine original train and test data vertically
# as previous Open Prices are not present in test dataset
# e.g. for predicting Open price for first date in test data, we will need stock open prices on 60 previous dates  
combine = pd.concat((train_data['Open'], test_data['Open']), axis = 0)
# our test inputs also contains stock open Prices of last 60 dates (as described above)
test_inputs = combine[len(combine) - len(test_data) - 60:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)


# In[ ]:


test_data.shape


# In[ ]:


# same steps as we followed while processing training data
X_test = []
for i in range(60, 480):
    X_test.append(test_inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
# inverse_transform because prediction is done on scaled inputs
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# ## Visualising the results

# In[ ]:


plt.plot(real_stock_price, color = 'red', label = 'Real APPL Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted APPL Stock Price')
plt.title('APPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('APPL Stock Price')
plt.legend()
plt.show()


# ### Hurray!! We made it. You can try with more than one indicators, not just one (in our case its 'Open'). But remember one thing these indicators /factors should be something that can influence APPL open stock price.
