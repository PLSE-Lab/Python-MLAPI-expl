# basic library
import numpy as np
import pandas as pd

# deep learning library
import theano
print("Theano:", theano.__version__)

import tensorflow
print("Tensorflow:", tensorflow.__version__)

import keras
print("Keras:", keras.__version__)

from keras.models import Sequential
from keras.layers import Dense

# model selection library
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

# load dataset
datasets = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        datasets.append(os.path.join(dirname, filename))
        #print(os.path.join(dirname, filename))

print("Dataset:",datasets[0])


# Load data
dataset = np.loadtxt(datasets[0], delimiter=",")

X = dataset[:, 0:8]; Y=dataset[:,8]

# define model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X, Y, epochs=150, batch_size=10)
# model evaluate
scores = model.evaluate(X, Y)
# show accuracy
print(model.metrics_names[1], scores[1]*100)

# -------------------------------- #
# TRAIN TEST SPLIT SELECTION
# -------------------------------- #

# split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# fit model
model.fit(X_train, y_train, epochs=150, batch_size=10)
# evaluasi model
scores = model.evaluate(X_train, y_train)
# show accuracy
print(model.metrics_names[1], scores[1]*100)
# prediksi
predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]
print("Akurasi prediksi:", (rounded == y_test).mean())

# -------------------------------- #
# 10-KFold SELECTION
# -------------------------------- #

n_spl = 10
k_fold = KFold(n_splits=n_spl) 

tot_akurasi = 0
tot_prediksi = 0
for k, (train, test) in enumerate(k_fold.split(X, Y)):
    model.fit(X[train], Y[train], epochs=150, batch_size=10)
    scores = model.evaluate(X[train], Y[train])
    akurasi = scores[1]*100
    tot_akurasi = tot_akurasi + akurasi
    predictions = model.predict(X[test])
    rounded = [round(x[0]) for x in predictions]
    tot_prediksi = tot_prediksi + (rounded == Y[test]).mean()

print(model.metrics_names[1], tot_akurasi / n_spl)
print("Akurasi prediksi:", tot_prediksi / n_spl)

# -------------------------------- #
# LOO SELECTION
# -------------------------------- #

loo = LeaveOneOut()

total_data = len(X)
tot_akurasi = 0
tot_prediksi = 0
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model.fit(X_train, y_train, epochs=10, batch_size=10)
    scores = model.evaluate(X_train, y_train)
    akurasi = scores[1]*100
    tot_akurasi = tot_akurasi + akurasi
    predictions = model.predict(X_test)
    rounded = [round(x[0]) for x in predictions]
    tot_prediksi = tot_prediksi + (rounded == y_test).mean()

print(model.metrics_names[1], tot_akurasi / total_data)
print("Akurasi prediksi:", tot_prediksi / total_data)

print("FINISH________SELESAI")

