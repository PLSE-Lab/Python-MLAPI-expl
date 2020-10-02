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
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Dataset; 28x28 images of handwritten digits 0-9
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Split data into feature matrix & target vector
X_tr = train.iloc[:,1:]
y_tr = train.iloc[:,0]

X_tr = tf.keras.utils.normalize(np.matrix(X_tr))
y_tr = keras.utils.np_utils.to_categorical(y_tr.values)

# Split into train/val sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size = 0.2, random_state = 42)

# Instantiate model
mnist_model = Sequential()

mnist_model.add(Dense(128, activation = "relu", input_shape = (X_train.shape[1],)))
mnist_model.add(Dense(128, activation = "relu"))

# Output layer
mnist_model.add(Dense(10, activation = "softmax"))

# Compile model
mnist_model.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ["accuracy"])

# Fit model
mnist_model.fit(X_train, y_train, epochs = 10)

# Model validation
val_loss, val_acc = mnist_model.evaluate(X_val, y_val)

# Test data
X_te = test
X_te = tf.keras.utils.normalize(np.matrix(X_te))

# Model predictions
predictions = mnist_model.predict(X_te)
p = [np.argmax(i) for i in predictions]

all_ids = list(range(1,len(p)+1))

submission = pd.DataFrame()
submission["ImageID"] = all_ids
submission["Label"] = p
submission.head()

# Save predictions
# submission.to_csv("mnist_submission.csv", index = False)

# Save model
# mnist_model.save('mnist.model')