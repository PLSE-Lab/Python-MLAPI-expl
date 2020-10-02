import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from keras.layers import *

train_data = pd.read_csv('../input/train.csv')

X, y = train_data.drop(['label'], axis=1) / 255,  train_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = keras.Sequential([
     keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
     keras.layers.Conv2D(24, kernel_size=(3,3), padding="same", activation='relu'),
     keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     keras.layers.Conv2D(8, kernel_size=(3,3), padding="same", activation='relu'),
     keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     keras.layers.Flatten(),
     keras.layers.Dense(1024, activation='relu'),
     keras.layers.Dropout(rate = 0.2 ),
     keras.layers.Dense(512, activation='relu'),
     keras.layers.Dropout(rate = 0.2 ),
     keras.layers.Dense(10),
     keras.layers.Activation("softmax")
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

X_test =  pd.read_csv('../input/test.csv') / 255
predictions = model.predict_classes(X_test)
pd.DataFrame({
    "ImageId": list(range(1,len(predictions)+1)), 
    "Label": predictions
}).to_csv("submission.csv", index=False, header=True)


