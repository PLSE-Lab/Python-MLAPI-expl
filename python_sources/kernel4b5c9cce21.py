# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_dir='/kaggle/input/sign-language-mnist/sign_mnist_train.csv'
test_dir='/kaggle/input/sign-language-mnist/sign_mnist_test.csv'

def get_data(fileName):
    with open(fileName) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        tmp_images=[]
        tmp_labels=[]
        first_line=True
        for row in csv_reader:
            if first_line:
                first_line=False
            else:
                tmp_labels.append(row[0])
                image_data = row[1:785]
                image_from_array = np.array_split(image_data,28)
                tmp_images.append(image_from_array)
        images=np.array(tmp_images).astype('float')
        labels=np.array(tmp_labels).astype('float')
    return images, labels

train_images,train_labels = get_data(train_dir)
test_images,test_labels = get_data(test_dir)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_images = np.expand_dims(train_images,axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size=20
)

validation_generator = validation_datagen.flow(
    test_images,
    test_labels,
    batch_size=20
)

print(train_images.shape)
print(test_images.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(26,activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

history = model.fit(
    train_datagen.flow(train_images,train_labels, batch_size=20),
    steps_per_epoch=100,
    epochs=25,
    validation_data = validation_datagen.flow(test_images,test_labels,batch_size=20),
    validation_steps=50
)

print(model.evaluate(test_images, test_labels))

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
