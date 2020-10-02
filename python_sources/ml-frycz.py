import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
X = train.values[:,1:].astype("float32")
Y = train.values[:,0].astype("int32")

print(len(Y))

cut = int(0.8*len(X))
X_train = X[:cut]
X_test = X[cut:]
Y_train = Y[:cut]
Y_test = Y[cut:]

num_pixels = X_train.shape[1]

print(X_train.shape)

X_train /= 255
X_test /= 255

print(Y_train)
print(Y_train.shape)

Y_train = np_utils.to_categorical(Y_train) # one-hot
Y_test = np_utils.to_categorical(Y_test) # one-hot
num_classes = Y_test.shape[1]

print(Y_train)
print(Y_train.shape)

def basic_model():
    model = Sequential()
    model.add(Dense(int(num_pixels*1.5), input_dim=num_pixels, kernel_initializer="normal", activation="relu"))
    model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = basic_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, Y_test)
print(scores)

plt.imshow(X_test[0].reshape(28, 28))
plt.savefig("test.png")
print(Y_test[0])

p = model.predict(X_test[112].reshape(1,784))
plt.imshow(X_test[112].reshape(28, 28))
plt.savefig("test112.png")
print(p)