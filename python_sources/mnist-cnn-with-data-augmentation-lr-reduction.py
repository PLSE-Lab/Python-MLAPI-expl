from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

EPOCHS = 35
BATCH_SIZE = 64
IMAGE_SHAPE = 28

#Load data
training_file = "../input/train.csv"
testing_file = "../input/test.csv"
training_data = pd.read_csv(training_file)
testing_data = pd.read_csv(testing_file)

#Resize and normalise
training_images = training_data.drop(['label'],axis=1)
training_labels = training_data['label']

num_training_images = training_images.shape[0]

train_image_vals = training_images.values.astype('float32')
train_label_vals = training_labels.values.astype('int32')
test_image_vals = testing_data.values.astype('float32')

train_image_vals = train_image_vals / 255.0
test_image_vals = test_image_vals / 255.0
train_image_vals = train_image_vals.reshape(train_image_vals.shape[0],28,28,1)
test_image_vals = test_image_vals.reshape(test_image_vals.shape[0],28,28,1)

#Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_image_vals, train_label_vals, test_size = 0.1, random_state=2)

#CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Conv2D(64, (5,5), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (5,5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#Learning reduction callback function
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                                    patience=3, 
                                                    verbose=1, 
                                                    factor=0.5, 
                                                    min_lr=0.00001)

model.summary()

#Image augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(x_train)

#Train model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
                              epochs=EPOCHS,
                              steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
                              validation_data=(x_val,y_val),
                              callbacks=[lr_reduction])

#Visualise our results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./training-data.png')
plt.show()


#Run prediction on test data and save results
results = model.predict(test_image_vals)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)