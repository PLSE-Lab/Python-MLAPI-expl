# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import random
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
def to_datetime(item):
    return datetime.datetime.strptime(item, '%Y-%m-%d')
def num_days_since_0122(item):
    return (item - to_datetime("2020-01-22")).days
def get_sequential_data(data_frame, time_step):
    inputs_c, inputs_f, inputs_geo, targets_c, targets_f = [], [], [], [], []
    for i in range(len(data_frame) - time_step - 1):
        if data_frame.iloc[i]['Country_Region'] == data_frame.iloc[i + time_step]['Country_Region']\
                and data_frame.iloc[i]['Province_State'] == data_frame.iloc[i + time_step]['Province_State']:
            inputs_c.append(np.array(data_frame.iloc[i: i + time_step][['ConfirmedCases_scaled']]).tolist())
            inputs_f.append(np.array(data_frame.iloc[i: i + time_step][['Fatalities_scaled']]).tolist())
            inputs_geo.append(np.array(data_frame.iloc[i: i + time_step][['Geo_Id']]).tolist())
            targets_c.append(np.array(data_frame.iloc[i + time_step][['ConfirmedCases_scaled']]).tolist())
            targets_f.append(np.array(data_frame.iloc[i + time_step][['Fatalities_scaled']]).tolist())
    return inputs_c, inputs_f, inputs_geo, targets_c, targets_f
if __name__ == "__main__":
    week = 3
    time_step = 5
    num_outputs = 1
    batch_size = 64
    epochs = 100
    learning_rate = 0.005
    validation_split = 0.
    kaggle_path = "/kaggle/input/covid19-global-forecasting-week-%s/"%(week)
    train_data = pd.read_csv(os.path.join(kaggle_path, "train.csv"))
    test_data = pd.read_csv(os.path.join(kaggle_path, "test.csv"))
    train_data = train_data.fillna({'Province_State': 'Unknown'})
    test_data = test_data.fillna({'Province_State': 'Unknown'})
    train_data["Date_dt"] = train_data["Date"].map(to_datetime)
    train_data["Day"] = train_data["Date_dt"].map(num_days_since_0122)
    test_data["Date_dt"] = test_data["Date"].map(to_datetime)
    test_data["Day"] = test_data["Date_dt"].map(num_days_since_0122)
    scaler_c = MinMaxScaler(feature_range=(0, 1))
    train_data['ConfirmedCases_scaled'] = scaler_c.fit_transform(train_data[['ConfirmedCases']])
    scaler_f = MinMaxScaler(feature_range=(0, 1))
    train_data['Fatalities_scaled'] = scaler_f.fit_transform(train_data[['Fatalities']])
    train_data['Geo_Id'] = train_data['Country_Region'].astype(str) + '_' + train_data['Province_State'].astype(str)
    encoder = LabelEncoder()
    train_data['Geo_Id'] = encoder.fit_transform(train_data['Geo_Id'])
    train_inputs, train_inputs_f, train_inputs_geo, train_targets_c, train_targets_f = get_sequential_data(train_data, time_step)
    indices_path = "indices.json"
    train_indices, validation_indices = [], []
    indices = [i for i in range(len(train_inputs))]
    random.shuffle(indices)
    validation_indices = indices[:int(len(train_inputs) * validation_split)]
    train_indices = indices[int(len(train_inputs) * validation_split):]
    validation_inputs = [train_inputs[i] for i in validation_indices]
    validation_inputs_f = [train_inputs_f[i] for i in validation_indices]
    validation_targets_c = [train_targets_c[i] for i in validation_indices]
    validation_targets_f = [train_targets_f[i] for i in validation_indices]
    temp_train_inputs = [train_inputs[i] for i in train_indices]
    temp_train_inputs_f = [train_inputs_f[i] for i in train_indices]
    temp_train_inputs_geo = [train_inputs_geo[i] for i in train_indices]
    temp_train_targets_c = [train_targets_c[i] for i in train_indices]
    temp_train_targets_f = [train_targets_f[i] for i in train_indices]
    train_inputs = temp_train_inputs
    train_inputs_f = temp_train_inputs_f
    train_inputs_geo = temp_train_inputs_geo
    train_targets_c = temp_train_targets_c
    train_targets_f = temp_train_targets_f
    input_shape = np.array(train_inputs).shape[-2:]
    print(train_inputs[0])
    def get_model(inputs = None, targets = None, validation_inputs = None, validation_targets = None):
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(32))
        model.add(Dropout(0.5))
        model.add(Dense(num_outputs, activation='relu'))
        model.compile(optimizer='adam', loss=tf.keras.losses.MSLE,
                          metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,validation_data=(validation_inputs, validation_targets))
        scores = model.evaluate(inputs, targets)
        print("Accuracy: {}".format(scores))
        return model
    case_model = get_model(inputs=train_inputs,
                           targets=train_targets_c,
                           validation_inputs=validation_inputs,
                           validation_targets = validation_targets_c)
    fatality_model = get_model(inputs=train_inputs_f,
                            targets=train_targets_f,
                            validation_inputs=validation_inputs_f,
                            validation_targets=validation_targets_f)
    test_data['ConfirmedCases'] = None
    test_data['Fatalities'] = None
    start = time.time()
    for i in range(len(test_data)):
        current = test_data.iloc[i]
        geo_df = train_data[(train_data.Country_Region == current.Country_Region) & (
                train_data.Province_State == current.Province_State)]
        hist_df = geo_df[(geo_df.Day >= (current.Day - time_step)) & (geo_df.Day <= (current.Day - 1))]
        if not hist_df.empty and hist_df.shape[0] == time_step:
            pred = case_model.predict(np.array(hist_df[['ConfirmedCases_scaled']]).reshape(1, input_shape[0], input_shape[1]))
            pred_f = fatality_model.predict(np.array(hist_df[['Fatalities_scaled']]).reshape(1, input_shape[0], input_shape[1]))
            inversed = scaler_c.inverse_transform(np.array(pred).reshape(-1, 1))
            inversed_f = scaler_f.inverse_transform(np.array(pred_f).reshape(-1, 1))

            test_data.loc[i, 'ConfirmedCases'] = inversed[0][0]
            test_data.loc[i, 'Fatalities'] = inversed_f[0][0]

            df = geo_df[geo_df.Day == current.Day]
            if df.empty:
                train_data = train_data.append({'ConfirmedCases': inversed[0][0], 'Fatalities': inversed_f[0][0],
                                                    'ConfirmedCases_scaled': pred[0][0],
                                                    'Fatalities_scaled': pred_f[0][0], \
                                                    'Day': current.Day, 'Country_Region': current.Country_Region,
                                                    'Province_State': current.Province_State}, ignore_index=True)
            if i > 0 and i % 100 == 0:
                progress = float(i) / float(len(test_data))
                print("Progress: %.2f%% Estimated time:%.0fs" % (progress * 100, (time.time() - start) / progress))
        test_data[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)