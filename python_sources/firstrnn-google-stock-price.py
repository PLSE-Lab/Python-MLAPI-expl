# predict the stock price of Google in 2017

# Part 1 - Data Preprocessing

# import libraries and dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset_train = pd.read_csv('../input/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling for the train data

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output;
# Timesteps means we use how many timesteps to predict the t(like if we want to predict
# 60, we need to use 0 to 50, and the timesteps are 60)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the input data in order to fit the keras layer

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Intializing the RNN
regressor = Sequential()

# Adding the First LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True,
                   input_shape = ( X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the Second LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the Third LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the Fourth LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compile the layers to the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



## Part 3 - Making predictions and visualize the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('../input/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price: Since every pridcted price for 2017 will need 60previous
# price, we need to concate the trainning and test set

dataset_total  = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# Now we want ensure the downbound and upbound of the input since it requires previous 60
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
  
X_test = np.array(X_test)

# reshape the test data to 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)