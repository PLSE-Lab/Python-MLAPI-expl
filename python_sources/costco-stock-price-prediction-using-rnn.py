#!/usr/bin/env python
# coding: utf-8

# As an active investor and a professional working in data science & AI, I decided to build a LSTM (Long Short Term Memory) model to see how it predicts the prices of the stocks in my portfolio. Costco Wholesales is my first candicate for modelling. I am using the 5-year COST closing prices from 2014 to 2018 as the train set, and the closing prices from 2019 to March 27th, 2020 as the test set. The data source is Yahoo Finance. 

# Import the train set. The train set includes stock price data from January 2nd, 2014 to December 28th, 2018. 

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('../input/COST Train.csv')
print(dataset_train)


# In[ ]:


# Selecting closing prices
training_set = dataset_train.iloc[:,4:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# I am using prices in the past 120 trading days or 6 months to predict the next day's stock price. Thus, the input variable X will be every 120 closing prices, and the output variable Y will be the very 121st closing price. 

# In[ ]:


# Creating a data structure with 120 timesteps and 1 output

X_train = []
y_train = []
for i in range(120, 1258):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Build the LSTM model. I chose 4 layers, 60 units of memory cells (since stock price movements are quite complicated), and a common dropout rate of 20%. 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding four LSTM layer and Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# The next step is to predict the test set.  

# In[ ]:


dataset_test = pd.read_csv('../input/COST Test.csv')
print(dataset_test)


# Add the last 120 observations (the last 6 months of 2018) from the train set to the test set, as they are the timesetps that must be included in predicting the first three months of 2019.

# In[ ]:


real_stock_price = dataset_test.iloc[:,4:5].values

# Combing the last 120 prices from train set with the test set 
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0) 
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)

# Applying the same data processing 
inputs = sc.transform(inputs) 
X_test = []
for i in range(120, 432):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Predict using the test data and compare with the actual close prices. 

# In[ ]:


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

compare = pd.DataFrame(predicted_stock_price).join(dataset_test['Close']) 
compare.rename(columns={0: "Predicted", "Close": "Actual"})


# Now I will visualize the comparison between all actual prices and predict prices. 

# In[ ]:


# Visualizing the results 
plt.plot(real_stock_price, color = 'red', label = 'Actual Costco Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Costco Stock Price')
plt.title('Costco Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Costco Stock Price')
plt.legend()
plt.show()


# As shown above, the model is getting the correct direction of the stock movement, although the gap becomes wider starting in around June 2019 and the model tends to underestimate the stock value and lags behind a bit.

# In[ ]:


# Evaluating 
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)


# Finally, I would like to take a look what the stock price will be on the next trading day according to the model. 

# In[ ]:


# Predicting the stock price in the next 120 days 

new = pd.DataFrame(columns=['Close'],index=[0])
newdays = pd.concat((dataset_test['Close'],new['Close']), axis = 0) 
newdays = newdays[len(newdays) - 1 - 120:].values
newdays = newdays.reshape(-1,1)
newdays = sc.transform(newdays) 
X_pred = []

for i in range(120, 121):
    X_pred.append(newdays[i-120:i, 0]) 
X_pred = np.array(X_pred)
X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
new_stock_price = regressor.predict(X_pred)
new_stock_price = sc.inverse_transform(new_stock_price) 

print("The price of the next trading day will be: $", new_stock_price)


# To improve the model, I can create more layers or add more units to each LSTM layer, which will increase the computation time. 

# Model & code reference: https://www.udemy.com/course/deeplearning/
