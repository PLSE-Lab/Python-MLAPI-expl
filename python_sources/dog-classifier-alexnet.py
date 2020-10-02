#!/usr/bin/env python
# coding: utf-8

# # Intro
# In this notebook, I build a simple convolution neural network (AlexNet) to classify dog breeds. The data consists of 20580 photos from 120 dog breeds. I'm mostly using Keras API with TensorFlow backend.
# 
# I've also built a VGG model notebook (https://www.kaggle.com/rexfwang/dog-classifier-vgg16) and VGG/ResNet transfer learning notebook (https://www.kaggle.com/rexfwang/dog-classifier-transfer-learning) using the same dataset. 

# # Data augmentation
# * Rescale the pixel range from 0-255 to 0-1
# * Random rotation < 20 degree
# * Sample-wise normalization -> zero mean and unit variance
# * Recalse the photo resolution to 227 x 227
# * Batch size 128

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import random
from sklearn import metrics


# In[ ]:


dataGen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.2)

train_data = dataGen.flow_from_directory(
    '../input/images/Images',
    target_size=(227, 227),
    class_mode='categorical',
    batch_size=128,
    subset='training')

validation_data = dataGen.flow_from_directory(
    '../input/images/Images',
    target_size=(227, 227),
    class_mode='categorical',
    batch_size=128,
    subset='validation')


# In[ ]:


def remove_prefix(name):
    idx = name.find("-")
    return name[idx+1:]

class_name = {v: remove_prefix(k) for k, v in train_data.class_indices.items()}
num_train = train_data.n
num_validation = validation_data.n


# # Sample inputs
# Randomly draw 5 samples from the first training batch and print them out

# In[ ]:


samples = 5
sample_idices = random.sample(range(32), samples)
plt.figure(figsize=(20,10))
for i in range(samples):
    idx = sample_idices[i]
    img = train_data[0][0][idx] # the 0 batch, feature, sample idx
    label = train_data[0][1][idx] # the 0 batch, label, sample idx
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    lable_idx = np.argmax(label)
    plt.xlabel(class_name[lable_idx])
plt.show()


# # Build CNN model
# - AlexNet: 5 conv layers, 2 fully connected layer, 1 softmax output layer
# - Batch normalization, max pooling and dropout layers in between
# 

# In[ ]:


model = keras.Sequential()

# 1st Convolutional Layer
model.add(keras.layers.Conv2D(96, 
                              input_shape=(227, 227, 3), 
                              kernel_size=(11,11), strides=(4,4), activation='relu'))
# Max Pooling
model.add(keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(keras.layers.BatchNormalization())

# 2nd Convolutional Layer
model.add(keras.layers.Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu'))
# Max Pooling
model.add(keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(keras.layers.BatchNormalization())

# 3rd Convolutional Layer
model.add(keras.layers.Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu'))

# 4th Convolutional Layer
model.add(keras.layers.Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu'))

# 5th Convolutional Layer
model.add(keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
# Max Pooling
model.add(keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Passing it to a Fully Connected layer
model.add(keras.layers.Flatten())
# 1st Fully Connected Layer
model.add(keras.layers.Dense(4096, activation='relu'))
# Add Dropout to prevent overfitting
model.add(keras.layers.Dropout(0.5))

# 2nd Fully Connected Layer
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))

# Output Layer
model.add(keras.layers.Dense(120, activation='softmax'))

model.compile(optimizer = keras.optimizers.Adam(0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# ## Train
# - 30 epochs
# - Accuracy and loss plots across epochs

# In[ ]:


history = model.fit_generator(train_data, epochs=30, validation_data=validation_data, workers=8)


# In[ ]:


history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(epochs)
plt.show()

print(pd.DataFrame(history_dict))


# # Evaluate validation set
# - Load the whole validation set
# - Classify them with the model

# In[ ]:


x_validation = np.ndarray([num_validation,227,227,3])
y_validation = np.ndarray([num_validation])

sampleIdx = 0;
num_batch = len(validation_data)
for i in range(num_batch):
    samples, labels = validation_data[i]
    count = labels.shape[0]
    labels = np.argmax(labels, axis=1)
    x_validation[sampleIdx:sampleIdx+count,:,:,:] = samples
    y_validation[sampleIdx:sampleIdx+count] = labels
    sampleIdx = sampleIdx + count


# In[ ]:


y_predict_softmax = model.predict(x_validation)
y_predict = np.argmax(y_predict_softmax, axis=1)


# # Sample prediction results

# In[ ]:


samples = 5
sample_idices = random.sample(range(32), samples)
plt.figure(figsize=(20,10))
for i in range(samples):
    idx = sample_idices[i]
    img = x_validation[idx]
    label = y_validation[idx]
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    pred = y_predict_softmax[idx]
    pred_classes = np.argsort(pred)
    pred_classes = np.flip(pred_classes)
    pred_classes = pred_classes[0:3]
    pred_prob = np.sort(pred)
    pred_prob = np.flip(pred_prob)
    pred_prob = pred_prob[0:3]*100
    xlabel = "Trush: {}\n".format(class_name[label])         +"{0}:{1:.1f}%\n".format(class_name[pred_classes[0]],pred_prob[0])         +"{0}:{1:.1f}%\n".format(class_name[pred_classes[1]],pred_prob[1])         +"{0}:{1:.1f}%\n".format(class_name[pred_classes[2]],pred_prob[2])
    plt.xlabel(xlabel)
plt.show()


# # Confusion matrix

# In[ ]:


conf_matrix = metrics.confusion_matrix(y_validation, y_predict)
plt.figure()
plt.imshow(conf_matrix)
plt.xlabel("Prediction")
plt.ylabel("True label")
plt.colorbar()


# # Conv filters

# In[ ]:


# conv1 filter weights
[kernel, _] = model.layers[0].get_weights()
plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(kernel[:,:,0,i],cmap="gray")

# conv2 filter weights
[kernel, _] = model.layers[3].get_weights()
plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(kernel[:,:,0,i],cmap="gray")
    
# conv3 filter weights
[kernel, _] = model.layers[6].get_weights()
plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(kernel[:,:,0,i],cmap="gray")


# # Layer Activations

# In[ ]:


# Build a new model to get the activation maps
layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_validation[0:1])


# In[ ]:


# Show input image
plt.figure()
plt.imshow(x_validation[0])
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.xlabel(class_name[y_validation[0]])

# Results from conv layers
activation_1st_conv = activations[0]
activation_2nd_conv = activations[3]
activation_3rd_conv = activations[6]

plt.figure(figsize=(20,5))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.imshow(activation_1st_conv[0,:,:,i], cmap="gray")

plt.figure(figsize=(20,5))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.imshow(activation_2nd_conv[0,:,:,i], cmap="gray")

plt.figure(figsize=(20,5))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.imshow(activation_3rd_conv[0,:,:,i], cmap="gray")

