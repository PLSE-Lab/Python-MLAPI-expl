# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:13:12 2018

@author: ASUS
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

#dataset=pd.read_csv("../input/diabetes.csv")
# load pima indians dataset
dataset = numpy.loadtxt('../input/diabetes.csv')
# split into input (X) and output (Y) variables
#train dataset x= training data
X = dataset[:,0:8]
#target of train dataset
Y = dataset[:,8]


model = Sequential()

model.add(Dense(units=16, input_dim=8, activation='relu'))
model.add(Dense(units=14, activation='relu'))
model.add(Dense(units=14, activation='relu'))
model.add(Dense(units=9, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=7, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=9, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, epochs=400, batch_size=10)


# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)
