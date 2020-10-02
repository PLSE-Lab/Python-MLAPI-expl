# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import math
import time
import datetime
import itertools
import keras
import h5py
import requests

from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import Adam, SGD, RMSprop

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


def normalize_data_frame(data_frame):
    min_max_scaler = preprocessing.MinMaxScaler()
    new_data_frame = data_frame.copy()
    new_data_frame['High'] = min_max_scaler.fit_transform(new_data_frame['High'].values.reshape(-1, 1))
    new_data_frame['Low'] = min_max_scaler.fit_transform(new_data_frame['Low'].values.reshape(-1, 1))
    new_data_frame['Open'] = min_max_scaler.fit_transform(new_data_frame['Open'].values.reshape(-1, 1))
    new_data_frame['Volume'] = min_max_scaler.fit_transform(new_data_frame['Volume'].values.reshape(-1, 1))
    new_data_frame['Adj Close'] = min_max_scaler.fit_transform(new_data_frame['Adj Close'].values.reshape(-1, 1))
    return new_data_frame


def denormalize(data_frame, normalized_values):
    min_max_scaler = preprocessing.MinMaxScaler()
    new_data_frame = data_frame.copy()
    temp = min_max_scaler.fit_transform(new_data_frame['Adj Close'].values.reshape(-1, 1))
    denormalize_values = min_max_scaler.inverse_transform(np.reshape(normalized_values, (-1, 1)))
    return denormalize_values


def prepare_training_and_testing_datasets(df, size):
    number_of_features = df.shape[1]
    bulk_data = []
    size = size + 1
    for i in range(len(df) - size):
        bulk_data.append(df[i:i + size])
    bulk_data = np.array(bulk_data)
    number_of_rows = int(round(0.9 * bulk_data.shape[0]))
    train = bulk_data[:number_of_rows]
    test = bulk_data[number_of_rows:]
    X_train = train[:, :-1]
    Y_train = train[:, -1][:, -1]
    X_test = test[:, :-1]
    Y_test = test[:, -1][:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], number_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], number_of_features))
    return [train, test, X_train, Y_train, X_test, Y_test]


def build_model(number_of_input_rows, number_of_input_columns):
    dropout_rate = 0.3
    model = Sequential()
    # model.add(GRU(256, input_shape=(number_of_input_rows, number_of_input_columns), return_sequences=True))
    # model.add(Dropout(dropout_rate))
    # model.add(LSTM(256, input_shape=(number_of_input_rows, number_of_input_columns), return_sequences=True))
    # model.add(Dropout(dropout_rate))
    model.add(LSTM(256, input_shape=(number_of_input_rows, number_of_input_columns), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(256, input_shape=(number_of_input_rows, number_of_input_columns), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
    start_time = time.time()
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005), metrics=['mean_squared_error'])
    print("Compilation Time : ", time.time() - start_time)
    print(model.summary())
    return model


def model_score(lstm_model, x_train, y_train, x_test, y_test, verbose):
    train_score = lstm_model.evaluate(x_train, y_train, verbose=0)
    test_score = lstm_model.evaluate(x_test, y_test, verbose=0)
    if bool(verbose):
        print('Train Score: %.5f MSE (%.2f RMSE)' % (train_score[0], math.sqrt(train_score[0])))
        print('Test Score: %.5f MSE (%.2f RMSE)' % (test_score[0], math.sqrt(test_score[0])))
    return train_score, test_score


def main():
    stock_data = pdr.DataReader('GOOG', 'yahoo', "2000-10-04", "2019-11-08")
    stock_data = stock_data.drop(columns=['Close'])
    print(stock_data.head())

    normalized_stock_data = normalize_data_frame(stock_data.copy())
    print(normalized_stock_data.head())

    # dat = stock_data.copy()
    dat = normalized_stock_data.copy()

    dat = dat.to_numpy()

    train, test, X_train, Y_train, X_test, Y_test = prepare_training_and_testing_datasets(dat, 20)
    model = build_model(X_train.shape[1], X_train.shape[2])
    model.fit(X_train, Y_train, batch_size=512, epochs=90, validation_split=0.1, verbose=1)

    # predictions = model.predict(X_test)

    normalized_predictions = model.predict(X_test)
    predictions = denormalize(stock_data, normalized_predictions)
    denormalized_Y_test = denormalize(stock_data, Y_test)

    model_score(lstm_model=model, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose=True)

    plt.plot(predictions, color='red', label='Prediction')
    # plt.plot(Y_test, color='blue', label='Actual')
    plt.plot(denormalized_Y_test, color='blue', label='Actual')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
