# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# Any results you write to the current directory are saved as output.

#Import
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

#Filepaths
traindata_path = "/kaggle/input/digit-recognizer/train.csv"
submdata_path = "/kaggle/input/digit-recognizer/test.csv"
samplesubmission_path = "/kaggle/input/digit-recognizer/sample_submission.csv"
#Define Model

model = Sequential()

model.add(Flatten(input_shape=(784,)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation ="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Prepare Data and Train Model
traindata = pd.read_csv("//kaggle//input//digit-recognizer//train.csv")
testdata = pd.read_csv("//kaggle//input//digit-recognizer//test.csv")/255.0

X = traindata.drop("label", axis=1)/255.0
y = to_categorical(traindata["label"], num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model.fit(X_train, y_train, epochs=25)

score = model.evaluate(X_test, y_test)

#Make prediction on Test data
pred_labels = model.predict(testdata)
pred_labels = np.argmax(pred_labels, axis=1)

submission = pd.DataFrame({"ImageId":range(1,len(pred_labels)+1), "Label":pred_labels})

#Output submission file
submission.to_csv("//kaggle//working//submission.csv", index=False)