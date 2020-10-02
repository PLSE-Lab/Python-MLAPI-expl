# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

dataset_train = pd.read_csv("/kaggle/input/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

print(dataset_train.info())
print(training_set)

#feature scalling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


#creating a data sructure  with 60 timesteps and 1 output
# which means for every 60 previous inputs, it will compute next output

X_train = []
y_train = []

for i in range(60,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)
print(y_train)


#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Building RNN

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))


#compile the RNN

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train,y_train,epochs=100,batch_size=32)


#Making predictions and visualizing the results

dataset_test = pd.read_csv("/kaggle/input/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

#getting the predicted stock price

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#visualizing the result
plt.plot(real_stock_price, color='red', label="Real Google Stock Price")
plt.plot(predicted_stock_price, color='blue', label="predicted stock price")
plt.title("Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






