'''
July23-2018 
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for cifar100 100-class image classification using loss sparse_categorical_crossentropy and 100 sigmoids in output
Results:
Neural Training On CPU: Time elapsed 00h:11m:46s
Neural Training On GPU: Time elapsed 00h:03m:06s

'''
import time
import tensorflow as tf
from keras.datasets import cifar100


(x_train, y_train),(x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

startTime = time.time()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(100, activation=tf.sigmoid)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test, y_test)

endTime = time.time()
#print elapsed time in hh:mm:ss format
hours, rem = divmod(endTime-startTime, 3600)
minutes, seconds = divmod(rem, 60)
print("Time elapsed: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))
print('finished running tf july23-2018 cifar100 multiclass image')