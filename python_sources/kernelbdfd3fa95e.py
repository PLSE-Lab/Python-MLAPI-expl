# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_data = pd.read_csv('../input/train.csv')
print(train_data.head())
train_x = train_data.drop(['PassengerId', 'Name', 'Survived', 'Ticket', 'Embarked', 'Cabin'], axis=1)
sex = {'male': -1,'female': 1}
train_x.Sex = [sex[item] for item in train_x.Sex]

train_x['Pclass'].fillna((train_x['Pclass'].mean()), inplace=True)
train_x['Sex'].fillna((train_x['Sex'].mean()), inplace=True)
train_x['Age'].fillna((train_x['Age'].mean()), inplace=True)
train_x['SibSp'].fillna((train_x['SibSp'].mean()), inplace=True)
train_x['Parch'].fillna((train_x['Parch'].mean()), inplace=True)
train_x['Fare'].fillna((train_x['Fare'].mean()), inplace=True)

train_y = train_data[['Survived']]

print(train_x.head())
print(train_y.head())

keras = tf.keras

scaler = StandardScaler()
scaler.fit(train_x)
poly = PolynomialFeatures()

train_x_value = poly.fit_transform(scaler.transform(train_x.values))
train_y_value = train_y.values

model = keras.Sequential([
    keras.layers.Dense(28, activation=tf.nn.leaky_relu, input_shape=(28,)),
    keras.layers.Dropout(0.8),
    keras.layers.Dense(28, activation=tf.nn.relu),
    keras.layers.Dense(7, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

batchSize = 40
epochs = 300
learningRate = 0.001
learning_Rate_Schedule = keras.optimizers.schedules.ExponentialDecay(learningRate, decay_steps=epochs, decay_rate=0.94)

optimizer = keras.optimizers.Adam(learning_rate=learning_Rate_Schedule)
loss = keras.losses.binary_crossentropy

model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#Train model
model.fit(train_x_value, train_y_value, validation_split=0.1, epochs=epochs, batch_size=batchSize)

trainingScores = model.evaluate(train_x_value, train_y_value, batch_size=batchSize)
print("Training %s: %.2f%%" % (model.metrics_names[1], trainingScores[1]*100))

# Test model
test_x_data = pd.read_csv('../input/test.csv')
test_x = test_x_data.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)
sex = {'male': -1,'female': 1}
test_x.Sex = [sex[item] for item in test_x.Sex]

test_x['Pclass'].fillna((test_x['Pclass'].mean()), inplace=True)
test_x['Sex'].fillna((test_x['Sex'].mean()), inplace=True)
test_x['Age'].fillna((test_x['Age'].mean()), inplace=True)
test_x['SibSp'].fillna((test_x['SibSp'].mean()), inplace=True)
test_x['Parch'].fillna((test_x['Parch'].mean()), inplace=True)
test_x['Fare'].fillna((test_x['Fare'].mean()), inplace=True)

scaler.fit(test_x)
test_x = poly.fit_transform(scaler.transform(test_x))

test_y_data = pd.read_csv('../input/gender_submission.csv')
test_y = test_y_data[['Survived']]

testingScores = model.evaluate(test_x, test_y, batch_size=batchSize)
print("Testing %s: %.2f%%" % (model.metrics_names[1], testingScores[1]*100))

y_pred = model.predict(test_x)
y_final = (y_pred > 0.5).astype(int).reshape(test_x.shape[0])

test_y_data['Survived'] = y_final
test_y_data.to_csv("submit.csv", index=False)
test_y_data.head()