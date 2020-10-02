# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from scipy import signal

print(os.listdir("../input"))

test_size = 800
num_classes = 5
#EMG reading device has num_sensors number of channels
num_sensors = 8
#num_sensors are read num_reads times for one prediction
num_reads = 6
sample_x_size = num_reads * num_sensors

def notch(data, val = 50, fs = 250):
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        fin = data = signal.lfilter(b, a, data)
    return fin
    
def bandpass(data, start = 5, stop = 124, fs = 250):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.lfilter(b, a, data, axis=0)

def read_data():
    allFiles = glob.glob("../input/*.txt")
    list = []
    aClass = 0
    for file in allFiles:
        df = pd.read_csv(file,index_col=None,skiprows=400+6,usecols=[1,2,3,4,5,6,7,8])
        read_matrix = np.asmatrix(df)
        print('Input', read_matrix.shape)
        read_matrix = np.apply_along_axis(notch, 0, read_matrix)[0]
        read_matrix = np.apply_along_axis(bandpass, 0, read_matrix)
        new_num_rows = read_matrix.shape[0] // num_reads
        #trimming #rows to multiple of num_reads
        read_matrix = read_matrix[:new_num_rows * num_reads:]
        #convert matrix to multiple reads of each sensor at a line
        read_matrix = read_matrix.reshape(new_num_rows, sample_x_size)
        roll = read_matrix
        list_rolls = []
        for i in range(1, num_reads - 1):
            roll = np.roll(read_matrix, -num_sensors)
            list_rolls.append(roll)
            
        for list_roll in list_rolls:
            #insert result class into the last column
            roll_with_class = np.insert(list_roll, sample_x_size, values=aClass, axis=1)
            list.append(roll_with_class)
        #insert result class into the last column
        read_matrix = np.insert(read_matrix, sample_x_size, values=aClass, axis=1)
        aClass+=1
        print('Samples: ', read_matrix.shape)
        print('Loaded class', read_matrix[0, sample_x_size])
        list.append(read_matrix)
    data = np.concatenate(list)
    values = np.asmatrix(data)
    np.random.shuffle(values)
    x_train = values[test_size:,:sample_x_size]
    y_train = values[test_size:,sample_x_size]
    x_test = values[:test_size,:sample_x_size]
    y_test = values[:test_size,sample_x_size]
    print('x_train ', x_train.shape)
    print('y_train ', y_train.shape)
    print('x_test ', x_test.shape)
    print('y_test ', y_test.shape)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = read_data()

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

model = Sequential()
# Dense(sample_x_size) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
model.add(Dense(40, activation='relu', input_dim=sample_x_size))
model.add(Dropout(0.2))
model.add(Dense(22, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=100)
score = model.evaluate(x_test, y_test, batch_size=test_size)
print(score)