import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

df = pd.read_csv('../input/BTC-INR.csv')
train = df.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train = sc.fit_transform(train)

X_train = train[0:1763]
y_train = train[1:1764]

X_train = np.reshape(X_train, (1763, 1, 1))

model = Sequential()
model.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(LSTM(units=4, activation='sigmoid'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=32, epochs=200)

X_test = train[1765:1824]
y_train = train[1766:1825]

X_test = sc.inverse_transform(X_test)
inputs = X_test
inputs = sc.transform(inputs)
inputs= np.reshape(inputs, (59, 1, 1))
predicted_stock_prices = model.predict(inputs)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices)

plt.plot(X_test, color='red', label='real_stock_price')
plt.plot(predicted_stock_prices, color='blue', label='predicted_stock_price')
plt.title('BTC-INR stock prices prediction')
plt.xlabel('Last 60 days prediction (April-May)')
plt.ylabel('BTC-INR stock price')
plt.legend()
plt.show()

# Evaluating the model
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(X_test, predicted_stock_prices))
print(rmse/500000) # 500000 avg BTC price 
