# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import os
import time
import warnings
import numpy as np
import pandas as pd
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window,datos_entrenar=0.9):

    if filename=='sp500.csv':
        f = open(filename, 'rb').read()
        data = f.decode().split('\n')
    
    else:    
    
        cierres=pd.read_csv(filename,usecols=[5])
        cierres.reset_index(inplace=True, drop=True)
        data=cierres.values

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
#        print(i)
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)
    
    result = np.array(result)

    print(datos_entrenar)
    row = round(datos_entrenar * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  
#    predecir.shape
    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

#Min_max_normalize
#if(data_min.empty): data_min = data.min()
#		if(data_max.empty): data_max = data.max()
#		data_normalised = (data-data_min)/(data_max-data_min)
        
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="binary_crossentropy", optimizer="rmsprop",metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
#    data=X_test
#    len(X_test)
#    window_size=50
#    prediction_len=50
    for i in range(int(len(data)/prediction_len)):
#        i=1
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
#            x=model.predict(curr_frame) si lo ponemos asi no funciona ya que el modelo
#            espera un array de 3 dimensiones y le damos solo 2 por ello usamos 
            curr_frame2=curr_frame[np.newaxis,:,:] # creamos curr_frame3 dimension
            x=model.predict(curr_frame2)
            x.shape # vemos que de un input de 3d (curr frame)
#           obtenemos un output 2d y solo queremos el valor inside 
            x=x[0,0] # extraemos valor inside)
            predicted.append(x) # añadimos valor inside a la lista de predicciones
            
#            Aqui empezamos con actualizacion de curr_frame
            
#            curr_frame.shape                     
            curr_frame = curr_frame[1:]  # lo que hacemos es selecciones todos menos el primero
#            haciendo esto se nos que da un curr_frame de len=49
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0) 
#            En el alinea de arriba lo que hacemos es basicamente insertar el ultimo valor predicho (predicted[-1])
#            en el ultimo lugar del array y de esa forma volvemos a obtener un array de len 50 y seguimos prediciendo
            
#            len(curr_frame)
        prediction_seqs.append(predicted)
    return prediction_seqs
    
def plot_results_multiple(predicted_data, true_data, prediction_len,divisa):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.title(divisa)
        plt.legend()
    plt.show()
global_start_time = time.time()    
epochs  = 2
seq_len = 100
X_train, y_train, X_test, y_test = load_data('../input/EURUSD1440.csv', seq_len,True, datos_entrenar=0.9)

model = build_model([1, 100,200,1]) # neurons=1, seq_len=25 es decir los valores que 
#    quiero que me predica
history=model.fit(X_train,y_train, batch_size=512,nb_epoch=epochs,validation_split=0.05)
plt.figure('model train vs validationloss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xticks(np.arange(len(history.history['loss'])))
#    plt.xticks(np.arange(len([history.history['loss']))
#    plt.plot(history.history['acc'])  
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()
predictions = predict_sequences_multiple(model,X_test, seq_len, 100) 
#	predicted = lstm_mito.predict_sequence_full(model, X_test, seq_len)
#	predicted = lstm_mito.predict_point_by_point(model, X_test)    
print(len(predictions[0]))
print('Training duration (s) : ', time.time() - global_start_time)
plot_results_multiple(predictions, y_test, 100,'usdjpy')
    #plot_results(predicted,y_test)