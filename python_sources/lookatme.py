import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import gc
from numba import jit
from IPython.display import display, clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pyarrow.parquet as pq
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
# replace '../input/signal/Copy of Book1(7786).xlsx' with the location of ur exel sheet
tr = pd.ExcelFile('../input/copy-of-book1/Book1.xlsx')
train_set = {sheet_name: tr.parse(sheet_name)
          for sheet_name in tr.sheet_names}
# ts = pd.ExcelFile('../input/signal/test.xlsx')
# test_set = {sheet_name: ts.parse(sheet_name) 
#           for sheet_name in ts.sheet_names}

X_train = []
X_train.append([])

y_train = []

j = 0;
i = 1;
for key in train_set:
    if key =="sheet1":
        continue  
    for value in train_set[key]['Amplitude ']:
        if i%56 == 0:
            j+=1
            X_train.append([])
        else:
            X_train[j] += [value]
        i+=1
    for value in train_set[key]['Type']:
        switcher = {
             "Sine Wave": 0,
             "Step Function": 1,
        }
        v = switcher.get(value, "nan")
        if v != 'nan':
            y_train.append(v)
y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
X_train,y_train = shuffle(X_train, y_train) 
print(y_train)

def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

n_signals = 1 #So far each instance is one signal. We will diversify them in next step
n_outputs = 1 #Binary Classification

#Build the model
verbose, epochs, batch_size = True, 20, 1
n_steps, n_length = 55, 1
X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
print(X_train.shape)
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

# for testing
# preds = model.predict(X_test)

# threshpreds = (preds>0.5)*1
# print(threshpreds)