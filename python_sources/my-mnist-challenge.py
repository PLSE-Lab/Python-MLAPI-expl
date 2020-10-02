# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Load files and get labels

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)

#Normalization 
X_train = X_train / 255
test = test/255


#Reshape
print(test.shape)
print(X_train.shape)

X_train = np.array(X_train, np.float32).reshape((-1,28,28,1))
test = np.array(test, np.float32).reshape((-1,28,28,1))
Y_train = np.array(Y_train, np.uint8)
print(X_train.shape)
print(test.shape)



print(X_train)
print(X_train.shape)

#train validation split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


#label encoding to cathegorical
from keras.utils import to_categorical

Y_cat_train = to_categorical(Y_train, num_classes = 10)
Y_cat_val = to_categorical(Y_val, num_classes = 10)

#define and train the model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER 
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.summary()

print(X_train.shape, Y_cat_train.shape)
model.fit(X_train,Y_cat_train,epochs=30)

model.evaluate(X_val,Y_cat_val)


from sklearn.metrics import classification_report

predictions = model.predict_classes(X_val)
print(classification_report(Y_val,predictions))
print(predictions)


predictions = model.predict_classes(test)
predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)

submission.to_csv("mnist_challenge.csv",index=False)
