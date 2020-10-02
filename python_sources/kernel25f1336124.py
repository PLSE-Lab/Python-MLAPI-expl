# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
import matplotlib.pyplot as plt


train_data = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

x_train, y_train = train_data, train_data[['label']].copy()
x_train.pop('label')

# Reshape dataframe
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)

# Create validation data
x_val = x_train[-10000:]
y_val = y_train[-10000:]

# Exclude validation data
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Scale pixel values from 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=5, verbose=1)

test_loss, test_acc = model.evaluate(x_val, y_val)

print("Loss: {}% \n"
      "Accuracy: {}%".format((test_loss * 100.0), (test_acc * 100.0)))

predictions = model.predict(x_test)
predictions = pd.DataFrame(predictions)

readable_predictions = []
for index, row in predictions.iterrows():
    readable_predictions.append(np.argmax(row))

submission['Label'] = readable_predictions

# Save submission
submission.to_csv('submission.csv', index=False)

model.save('Digit_Recognizerv1.model')