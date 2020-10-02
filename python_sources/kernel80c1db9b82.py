# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np
import pandas as pd

from keras import models
from keras import layers
from keras import optimizers
from keras import losses 
from keras import metrics

import os

def preparation(fileName):
    train = pd.read_csv(fileName)
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train["Age"] = train["Age"].fillna(int(train["Age"].mean()))

    train["Embarked"] = train["Embarked"].fillna("S")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2

    train = train.drop(["Name"], axis=1)
    train = train.drop(["Ticket"], axis=1)
    train = train.drop(["Cabin"], axis=1)

    return train

def preparationTrain(fileName):
    data = preparation(fileName)
    data = data.drop(["PassengerId"], axis=1)
    y = data['Survived'].copy()
    x = data.drop(['Survived'], axis=1)

    return x, y

def preparationTest(fileName):
    data = preparation(fileName)
    ids = data['PassengerId'].copy()
    x = data.drop(["PassengerId"], axis=1)
    
    return ids, x


def createModel():
    model = models.Sequential()
    model.add(layers.Dense(35, activation="relu", input_shape=(7,)))
    model.add(layers.Dense(35, activation="relu"))
    model.add(layers.Dense(25, activation="relu"))
    model.add(layers.Dense(17, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

    return model

xTrain, yTrain = preparationTrain("../input/train.csv")

validationSize = 150

xVal = xTrain[:validationSize]
xTrainPart = xTrain[validationSize:]

yVal = yTrain[:validationSize]
yTrainPart = yTrain[validationSize:]

model = createModel()

history = model.fit(xTrainPart, yTrainPart, epochs=50, validation_data=(xVal, yVal))

ids, xTest = preparationTest("../input/test.csv")
predictions = model.predict_classes(xTest)

new_output = ids.to_frame()
new_output["Survived"]=predictions
new_output.head(10)
new_output.to_csv("submission.csv",index=False)
