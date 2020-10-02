# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/red-electrica-espaola-consumo-y-precios/2020-03-26-15-33_chronograf_data.csv")
N = int(data.shape[0]/2)
demandas = data.iloc[:N]
precios = data.iloc[N:]

#Carga y generación de atributos
df = pd.DataFrame(columns = ['precio','demanda','precio_diff','demanda_diff'], index = data._time[:N].map( lambda t: pd.to_datetime(t)))
df.precio = precios._value.values
df.demanda = demandas._value.values
df.precio_diff = df.precio.pct_change()
df.demanda_diff = df.demanda.pct_change()
df = df[1:] #dropna

#Escalar con los valores del conjunto de entrenamiento.
original_train = df[df.index<pd.to_datetime('2019-01-01').tz_localize('Europe/Madrid')]
original_test = df[df.index>pd.to_datetime('2019-01-01').tz_localize('Europe/Madrid')]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(original_train)
df = scaler.transform(df)
df = pd.DataFrame(df, columns = ['precio','demanda','precio_diff','demanda_diff'])
# Transformar datos al formato de entrada y salida de las Redes neuronales recurrentes [N,H,F]  Siendo N el número de muestras, H el horizonte y F el número de atributos a utilizar.
input_window_length = 32
horizon = 24      

def dataset_slidding_window(X_dataset,horizon,forecast,num_features,output):
    num_samples = X_dataset.shape[0] - input_window_length - horizon
    X = np.zeros((num_samples,input_window_length,X_dataset.shape[1]))    
    Y = np.zeros((num_samples,input_window_length))
    for i in range(num_samples):
        subset = np.array(X_dataset.iloc[i:i+input_window_length,:num_features])
        X[i,:,:] = subset
        subset = np.array(X_dataset.iloc[i+input_window_length:i+input_window_length+horizon,output])
        Y[i,:] = subset
    return X,Y
    



X,Y = dataset_slidding_window(df,input_window_length,horizon,4,0)



X_train = X[:-31*horizon]
Y_train = Y[:-31*horizon]
X_test = X[-31*horizon:]
Y_test = Y[-31*horizon:]



from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]),return_sequences = False))
model.add(Dropout(0.5)),
model.add(Dense(25))
model.add(Dense(Y.shape[1]))

model.compile(loss='mae', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# fit network
history = model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_test, Y_test), verbose=2, shuffle=True, callbacks = [early_stopping,checkpoint])
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


prediction = model.predict(X_test)
plt.plot(prediction[30])
plt.plot(Y_test[30])


