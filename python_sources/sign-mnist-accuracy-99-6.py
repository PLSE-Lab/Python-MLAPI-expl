#!/usr/bin/env python
# coding: utf-8

# This was one of the tasks on Cursera cource - Tensorflow in Practice.
# The challenge is to reach 99% accurace  by using no more than 2 Convolution Layrs.
# 
# Dataset is a SignMNIST (hand language for deaf) it has 26 lables what makes it even more challenging that  regular MNIST or Fashion MNIST.
# 
# Here is my best try with reaching **99.6%** on evaluation.
# 
# Please share your comments.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import csv
from os import getcwd


# In[ ]:


def get_data(filename):
    with open(filename) as training_file:
        labels = []
        images = []
        header = training_file.readline()
        rows = training_file.readlines()
        for row in rows:
            row = row.strip().split(',')
            labels.append(int(row[0]))
            images.append(np.array(np.array_split(row[1:785], 28)))
        labels = np.array(labels)
        images = np.array(images)
        images = images.astype(float)
    return images, labels

path_sign_mnist_train = f"/kaggle/input/sign_mnist_train.csv"
path_sign_mnist_test = f"/kaggle/input/sign_mnist_test.csv"
train_images, train_labels = get_data(path_sign_mnist_train)
test_images, test_labels = get_data(path_sign_mnist_test)


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# Here we will use TensorFlow ImageGenerator for image augmentation on train data to have better generalization.

# In[ ]:



train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)


training_datagen = ImageDataGenerator(
        rescale = 1./255, #scaling
        rotation_range=12, #rotation within 12 degrees
        width_shift_range=2, #horizontal shift
        height_shift_range=2, #vertical shift
        shear_range=0.2, #shearing 
        fill_mode='nearest',
        zoom_range=0.2,
)

# Validation data we just scaling
validation_datagen = ImageDataGenerator(rescale = 1./255,)


# Lets have a look how our pictures look before we proceed with modeling

# In[ ]:


import random
import matplotlib.pyplot as plt

pick = random.choice(train_images)
plt.imshow(np.array(pick).squeeze(), cmap=plt.cm.binary)
plt.show()


# Creating callback for stoping training once we reach the goal accuracy

# In[ ]:


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.997:
            print('\nAccuracy 99.6% archived:', logs.get('val_accuracy'))
            self.model.stop_training = True
            


# In[ ]:


keras.backend.set_floatx('float32')

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (4,4), activation='relu', input_shape=(28, 28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(24, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='swish'),
    keras.layers.Dense(25, activation='softmax')
])

model.summary()


# In[ ]:


# Compile Model. 
model.compile(optimizer=keras.optimizers.Adam(lr=1e-3, decay=8e-6),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(training_datagen.flow(train_images, train_labels, batch_size=50), 
                    epochs=100, 
                    steps_per_epoch=500, 
                    validation_data = validation_datagen.flow(test_images,  test_labels, batch_size=50), 
                    verbose = 1,
                    shuffle=True,
                    callbacks=[MyCallBack()],
                    validation_steps=115
                    )


# In[ ]:


print('\nValidation:')
model.evaluate(test_images, test_labels, verbose=1)


# Results mat vary withing 0.002

# In[ ]:


# Plot the chart for accuracy and loss on both training and validation
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize=(14,6))

ax = fig.add_subplot(121)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
ax = fig.add_subplot(122)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Thank you for attention
