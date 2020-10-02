# This Python 3 environment comes with many helpful analytics libraries inst

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import time
import requests
import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
dataset = pd.read_csv('../input/kraken/kraken1015to9-10-19.csv') # load data from csv

dataset.drop('volume_currency', axis=1, inplace=True)
dataset.drop('Weighted_Price', axis=1, inplace=True)
dataset.drop('Unnamed: 0', axis=1, inplace=True)

#dataset["t"]=dataset["Timestamp"]
dataset = dataset.set_index('Timestamp')
#conv to --> per day
last = dataset.iloc[-1:, 0].index.values.item()

dataset_MAIN = pd.DataFrame()
#To set names of columns
dataset_MAIN.insert(0, 'Open', 1, allow_duplicates = False)
dataset_MAIN.insert(1, 'High', 1, allow_duplicates = False)
dataset_MAIN.insert(2, 'Low', 1, allow_duplicates = False)
dataset_MAIN.insert(3, 'Close', 1, allow_duplicates = False)
dataset_MAIN.insert(4, 'volume_btc', 1, allow_duplicates = False)
#dataset_MAIN.insert(5, 'Time', 1, allow_duplicates = False)

a = dataset.iloc[1:2, 0].index.values.item()
b = a+86400
#b = a+14400

while b <= last:
    data = dataset.loc[a:b]
    if data.shape[0] == 0:
        a = b
        b = b+86400
    else:
        Open = data.iloc[0:1, 0]
        Open = float(Open)
        Close = data.iloc[-1:, 3]
        Close = float(Close)
        Vol = data['volume_btc'].sum() 
        dataset_MAIN.loc[b] = Open, data['High'].max(), data['Low'].min() , Close, Vol
        a = b
        b = b+86400
        #To take data from 1-1-2017
dataset_MAIN = dataset_MAIN.iloc[413:,:]



dataset_TRAIN = dataset_MAIN.iloc[:int(len(dataset_MAIN) * 0.66), :]
dataset_TEST = dataset_MAIN.iloc[len(dataset_TRAIN):,:]

    #2) scaling
from sklearn.preprocessing import StandardScaler
scalerTRAINING = StandardScaler()
scalerTESTING= StandardScaler()


dataset_TRAIN_scaled = scalerTRAINING.fit_transform(dataset_TRAIN)
dataset_TEST_scaled = scalerTESTING.fit_transform(dataset_TEST) 
dataset_TRAIN_scaled.shape
def get_data(data, window_size):
    X = []
    y = []
    i = 0
    
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i += 1
    assert len(X) ==  len(y)
    return X, y

    #3) Reshape to muti-dimensions
x,y = get_data(dataset_TRAIN_scaled,window_size=100)
X_train = np.array(x[:len(x)])
y_train = np.array(y[:len(y)])
print(x)
y_train.shape
aTEST ,bTEST = get_data(dataset_TEST_scaled, window_size=100)
X_test = np.array(aTEST[:len(aTEST)])
Y_test = np.array(bTEST[:len(bTEST)])
Y_test.shape

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, LSTM, Dropout,GRU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import time
now = time.time()
print(now)
regressor = Sequential()
regressor.add(LSTM(units=1000, input_shape=(X_train.shape[1],5), activation="linear"))
regressor.add(Dropout(0.0155))
#regressor.add(LSTM(units=50, activation="relu"))
#regressor.add(Dropout(0.0424)), return_sequences=True
regressor.add(Dense(units=5))
#adam=optimizers.Adam(lr=0.0056)
regressor.compile(optimizer=Adam(), loss='mean_absolute_error')

#early_stopping = EarlyStopping(patience=0, verbose=1)
history=regressor.fit(X_train, y_train, epochs=300 , batch_size=72, validation_data=(X_test , Y_test))#, callbacks=[early_stopping])
TRAIN_loss=history.history['loss']
VALIDATE_loss=history.history['val_loss']
plt.plot(TRAIN_loss)
plt.plot(VALIDATE_loss)
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
now = time.time()
print(now)


resultPRED = regressor.predict(X_test)
print(resultPRED)
   #3)  EVALUATE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, confusion_matrix
from sklearn.metrics import average_precision_score
from math import sqrt

Acc = r2_score(Y_test, resultPRED) * 100
print('Accuracy = ', round(Acc, 2), '%', '\n')

MSE = mean_squared_error(Y_test, resultPRED)
print ('MSE = ', round(MSE, 4), '\n')

RMSE = sqrt(mean_squared_error(Y_test, resultPRED))
print ('RMSE = ', round(RMSE, 4), '\n')

MAE = mean_absolute_error(Y_test, resultPRED)
print ('MAE = ', round(MAE, 4), '\n')

MAPE = np.mean(np.abs(((Y_test - resultPRED) / Y_test)) * 100)
print ('MAPE = ', round(MAPE, 4), '\n')

print('precision_score=', average_precision_score(Y_test, resultPRED , average="macro"))


resultPRED = scalerTESTING.inverse_transform(resultPRED)
Y_test = scalerTESTING.inverse_transform(Y_test)

plt.plot(Y_test[:,3],color="b",label="real Close")
plt.plot(resultPRED[:,3],color="r",label="predicted Close")

plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.grid(True)
plt.show()

 
