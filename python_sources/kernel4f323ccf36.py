import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

train = np.array(train, dtype = 'float32')
test = np.array(test, dtype = 'float32')

x_train = train[:, 1:]/ 225
y_train = train[:,0]

x_test = test[:, 1:]/ 225
y_test = test[:,0]

x_train  = x_train.reshape(len(x_train),28,28,1)
x_test = x_test.reshape(len(x_test),28,28,1)

model = tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape=(28,28,1)),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Conv2D(64,(3,3),activation = tf.nn.relu),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(128,activation = tf.nn.relu),
                          keras.layers.Dense(10,activation = tf.nn.softmax)])


model.compile(optimizer = tf.train.AdamOptimizer(learning_rate=0.0001),loss = 'sparse_categorical_crossentropy',  metrics=['accuracy'])


model.fit(x_train,y_train, epochs =10 )
print("\nEvaluation:")
model.evaluate(x_test,y_test)

model.summary()






