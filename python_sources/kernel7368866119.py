# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adagrad

np.random.seed(1547)

# Open data csvs
titanicTrain = pd.read_csv("../input/titanic/cleanTitanicTrain.csv")
titanicTest = pd.read_csv("../input/titanic/cleanTitanicTest.csv")

# Columns used to predict
predictorNames = ["Pclass", 
                  "Sex", 
                  "Age",  
                  "Fare"]

myScaler = StandardScaler()
trainPredictors = myScaler.fit_transform(titanicTrain[predictorNames].values)
trainTarget = titanicTrain['Survived']

trainTargetonehot = pd.get_dummies(titanicTrain['Survived']).values
testPredictors = myScaler.transform(titanicTest[predictorNames].values)
testPIDs = titanicTest['PassengerId']

start = time()
myModel = Sequential()
myModel.add(Dense(10, 
                  input_dim = 4))
myModel.add(Dense(10))
myModel.add(Dense(10))
myModel.add(Dense(output_dim = 2))
myModel.add(Activation('relu'))
optimizer = Adagrad(lr = 0.01, epsilon=1e-08, decay=0.0)
myModel.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["binary_accuracy"])
myModel.fit(trainPredictors,
            trainTargetonehot)
myModelPredictions = myModel.predict_classes(trainPredictors)

print(classification_report(trainTarget,
                            myModelPredictions))

predictedTarget = myModel.predict_classes(testPredictors)
mySubmission = pd.DataFrame({'PassengerID': testPIDs,
                             'Survived': predictedTarget})
mySubmission.to_csv("mySubmission.csv",
                    index=False)