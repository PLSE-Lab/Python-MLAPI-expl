'''
Dated: July31-2018 
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for fashion mnist image browsing and predictions

Results:
test loss: 0.36 , test acc: 0.87
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

print("x_train.shape,y_train.shape")
print(x_train.shape,y_train.shape)
print('x_train')
print(x_train)

print("showing the 17th image in the dataset")
plt.imshow(x_train[16])
plt.show()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28 * 28,)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)

#ask the model to predict the classes of unclassified samples
predictions = model.predict(x_test)

print("predictions[16]")
print(predictions[16])

print("np.argmax(predictions[16])")
print(np.argmax(predictions[16]))

print("y_test[16]")
print(y_test[16])

