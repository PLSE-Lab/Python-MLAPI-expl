#!/usr/bin/env python
# coding: utf-8

# # Flower Image Classifier Model with Tensorflow Keras, Using a Deep Learning CNN Architecture.
# 
# Build from zero CNN Architecture with Tensorflow to identify 5 differents types of flowers.
# 
# Please, comment suggestions or doubts about the project.
# Hope you Like my first CNN :)

# # Step 1: Import Modules

# In[ ]:


#Use Local Archives
import os
import cv2

#Visualization and Manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Deep Learning Framework, CNN
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Manipulate Operations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import randint


# # Step 2: Read and Get Images Data/Labels

# In[ ]:


#Load, Read and Get Images
SIZE=100
data_path = "/kaggle/input/flowers-recognition/flowers/"

# lists to store data
data = []
label = []

#Folders/Labels in the Dataset
folders = os.listdir(data_path)

for folder in folders:
    for file in os.listdir(data_path + folder):
        if file.endswith("jpg"):
            img = cv2.imread(data_path + folder + '/' + file)
            img = cv2.resize(img, (SIZE,SIZE), interpolation = cv2.INTER_AREA)
            data.append(img)
            label.append(folder)
        else:
            continue


# # Step 3: Split Train/Test with Encoding Categories

# In[ ]:


#Transforming into numpy array
data = np.array(data)
label = np.array(label)

#Split Dataset into Train and Test sets
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=25)

#Get the Categories/Classes
label_categories = np.unique(label)
test_label_names = test_label


# In[ ]:


#Transforming object categories into numerical
encoder = LabelEncoder()

train_label = encoder.fit_transform(train_label).astype(int)
test_label = encoder.fit_transform(test_label).astype(int)


# # Step 4: Data Augmentation, improve Dataset to avoid Overfitting

# In[ ]:


datagen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.25,
                    height_shift_range=.25,
                    horizontal_flip=True,
                    shear_range=45.0,
                    zoom_range=0.5
                    )

datagen_valid = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.25,
                    height_shift_range=.25,
                    horizontal_flip=True,
                    shear_range=45.0,
                    zoom_range=0.5
                    )

datagen_train.fit(train_data)
datagen_valid.fit(test_data)


# # Step 5: Building CNN Model

# In[ ]:


#Deep Learning CNN Architecture
fclassifier = Sequential([
    Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(SIZE, SIZE,3)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='sigmoid')
])

#Model Summary Visualize
fclassifier.summary()


# In[ ]:


fclassifier.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])


# # Step 6: Training Model

# In[ ]:


checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
history = fclassifier.fit(datagen_train.flow(train_data,train_label, batch_size=32),
                         validation_data=datagen_valid.flow(test_data,test_label, batch_size=32),
                         callbacks=[checkpointer],
                         epochs=40)


# In[ ]:


fclassifier.load_weights('model.weights.best.hdf5')


# # Step 7: Visualize Training and Validation Accuracy/Loss

# In[ ]:


#Visualize Training and Validation Accuracy/Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(40)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# # Step 8: Visualize Some Classifications

# In[ ]:


#Visualize Some Image Classifications
fig = plt.figure(figsize=(15,10))
fig.suptitle("Some Classifications Example", fontsize=16)

predictions = fclassifier.predict(test_data)
predictions = np.argmax(predictions, axis=1)

for i in range(12):
    number = randint(0, len(predictions))
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(test_label_names[number])
    plt.imshow(test_data[number], cmap=plt.cm.binary)
    if test_label_names[number] == label_categories[predictions[number]]:
        plt.xlabel(label_categories[predictions[number]], color='green')
    else:
        plt.xlabel(label_categories[predictions[number]], color='red')
plt.show()

