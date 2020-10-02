# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras as keras

def getTestSet():
    df = pd.read_csv("/kaggle/input/titanic/test.csv")
    df = df.drop(labels=["Name", "Cabin", "Ticket", "Embarked", "PassengerId"], axis=1)
    df = df.fillna(df.mean())
    df = df.replace(to_replace={"male": 0, "female": 1})
    df=(df-df.min())/(df.max()-df.min())
    return df.values


def getDF(filePath):
    df = pd.read_csv(filePath)
    df = df.drop(labels=["Name", "Cabin", "Ticket", "Embarked", "PassengerId"], axis=1)
    df = df.fillna(df.mean())
    df = df.replace(to_replace={"male": 0, "female": 1})
    df=(df-df.min())/(df.max()-df.min())
    X = df[df.columns[1:]].values
    Y = df[df.columns[0]].values #only the first column
    return X, Y

train_X, train_Y = getDF("/kaggle/input/titanic/train.csv")
test_X = getTestSet()

model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_dim=6, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=10, batch_size=1)

_, accuracy = model.evaluate(train_X, train_Y)
print('Train Accuracy: %.2f' % (accuracy*100))
predictions = model.predict_classes(test_X)
df = pd.read_csv("/kaggle/input/titanic/test.csv")
df = df.drop(labels=["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
df["Survived"] = predictions
print(df)
df.to_csv("out.csv", index = None)
