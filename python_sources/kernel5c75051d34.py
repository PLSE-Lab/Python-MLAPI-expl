import numpy as np
import pandas as pd
from pprint import pprint

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
n_classes = 10

def process_data(data):
    y = keras.utils.all_utils.to_categorical(data.label, n_classes)
    x = data.values[:, 1:].reshape(data.shape[0], 28, 28, 1)
    return x / 255, y
train_x, train_y = process_data(train)

model = Sequential()
model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_x, train_y, batch_size=128, epochs=30, validation_split=0.2)

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
x = test.values.reshape(test.shape[0], 28, 28, 1)
predictions = np.argmax(model.predict(x), axis=1)

submission = pd.DataFrame({"ImageId": range(1, test.shape[0] + 1), "Label": predictions})
submission.to_csv("submission.csv", index=False)
print("Complete")