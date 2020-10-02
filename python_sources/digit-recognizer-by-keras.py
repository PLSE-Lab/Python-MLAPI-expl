# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
print('TensorFlow version: ', tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sample_submission = open('/kaggle/input/digit-recognizer/sample_submission.csv')
test = open('/kaggle/input/digit-recognizer/test.csv')
train = open('/kaggle/input/digit-recognizer/train.csv')

train = pd.read_csv(train)
y_train = train['label'].to_numpy()

#print(y_train)#label
y_train = pd.get_dummies(y_train)
pd.DataFrame(y_train)
#print(y_train.shape)#label

del train['label']
x_train = train.to_numpy()
#print(x_train.shape)
model = Sequential()
model.add(Dense(units=533, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))






model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=128)

test = pd.read_csv(test)

x_test = test.to_numpy()
#print(x_test.shape)
y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape)
model.save("model.h5")
np.savetxt("y_pred.csv", y_pred, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
a = open("y_pred.csv")
a = pd.read_csv(a, header=None)
a = a.to_numpy()

import csv
index = 1
with open('sample_submission.csv', 'w', newline='') as csvfile:
    fieldnames = ['ImageId', 'Label']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in a:
        writer.writerow({'ImageId': index, 'Label': np.argmax(i)})
        index = index + 1
    




#sample_submission.write(y_pred)
#sample_submission.close()

#score = model.evaluate(x_test, y_test, batch_size=128)
#print(score)
# Any results you write to the current directory are saved as output.