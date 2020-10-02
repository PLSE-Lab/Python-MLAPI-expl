# PlaidML test script: https://github.com/plaidml/plaidml
# 15.2.2019, Sakari Lukkarinen
# Helsinki Metropolia University of Applied Sciences

# Running this script with:
# GPU off: 31.95 seconds
# GPU on:   1.41 seconds
# Speed up: (31.91:1.41) = 22.6 times

# Remember to change Internet Connected from Settings !!!

#!/usr/bin/env python
import numpy as np
import os
import time

# os.environ["KERAS_BACKEND"] = "tensorflow"
# print(os.environ["KERAS_BACKEND"])

import keras
import keras.applications as kapp
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
np.savez('cifar10_data', x_train = x_train, y_train_cats = y_train_cats, x_test = x_test, y_test_cats = y_test_cats)

batch_size = 8
x_train = x_train[:batch_size]
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
model = kapp.VGG19()
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
y = model.predict(x=x_train, batch_size=batch_size, verbose = 0)

# Now start the clock and run 10 batches
print("Timing inference...")
start = time.time()
for i in range(10):
    y = model.predict(x=x_train, batch_size=batch_size, verbose = 0)
print("Ran in {} seconds".format(time.time() - start))