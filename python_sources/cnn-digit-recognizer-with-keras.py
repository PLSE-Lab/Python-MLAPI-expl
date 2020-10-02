# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

INPUT_DIR = "../input"

def read_data(file_name):
    # Read in data from csv file
    data = np.genfromtxt(os.path.join(INPUT_DIR, file_name + ".csv"),
                         skip_header=True,
                         delimiter=',', )

    x = data
    y = None

    # Separate input data from targets
    if file_name == 'train':
        x = data[:, 1:]
        y = data[:, 0]
        # Encode labels to one-hot encoding
        y = to_categorical(y)

    # Scale input data from 0-255 interval to 0-1 interval
    x = x / 255.0

    # Reshape data from (num_samples, 784) to (num_samples, 28, 28, 1)
    x = x.reshape((x.shape[0], 28, 28, 1))

    return x, y

# Load data
x_train, y_train = read_data('train')

# Separate validation and training data
# x_val = x_train[:5000]
# y_val = y_train[:5000]
# partial_x_train = x_train[5000:]
# partial_y_train = y_train[5000:]

# Create model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     batch_size=128,
#                     epochs=7,
#                     validation_data=(x_val, y_val))

history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=7)

print(history.history)

# Predict test data
x_test, _ = read_data('test')

predictions = model.predict(x_test)
predictions = [[index + 1, np.argmax(p)] for index, p in enumerate(predictions)]
predictions = np.array(predictions).astype(int)

np.savetxt('submission.csv', predictions, fmt='%i', delimiter=',', header='ImageId,Label')
    