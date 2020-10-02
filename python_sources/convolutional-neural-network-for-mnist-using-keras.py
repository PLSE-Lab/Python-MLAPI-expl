import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Prepare variables
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1) 
del train

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
test = test / 255.0

# Reshape image (height=28px, width=28px, canal=1)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# One hot encode outputs
num_classes = 10
Y_train = np_utils.to_categorical(Y_train, num_classes=num_classes)

# Define the model
def CNNmodel():
    # Create model
    model = Sequential()
    model.add(
        Conv2D(
            filters=30,
            kernel_size=(5, 5),
            input_shape=(28, 28, 1),
            activation='relu',
            padding='Same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            filters=15,
            kernel_size=(3, 3),
            activation='relu',
            padding='Same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build the model
model = CNNmodel()

# Fit the model
model.fit(X_train, Y_train, epochs=10, batch_size=200)

# Predict results
results = model.predict(test)
results = np.argmax(results,axis=1)
results = pd.Series(results,name='Label')

# Submit results to kaggle
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)
submission.to_csv('cnn_mnist_keyras.csv', index=False)
