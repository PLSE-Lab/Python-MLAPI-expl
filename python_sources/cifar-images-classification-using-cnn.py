#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AmitHasanShuvo/Machine-Learning-Projects/blob/master/CIFAR_Images_classification_using_CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Author: Kazi Amit Hasan <br>
# Department of Computer Science & Engineering, <br>
# Rajshahi University of Engineering & Technology (RUET) <br>
# Website: https://amithasanshuvo.github.io/ <br>
# Linkedin: https://www.linkedin.com/in/kazi-amit-hasan-514443140/ <br>
# Email: kaziamithasan89@gmail.com

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import datasets,models,layers


# In[ ]:


data = tf.keras.datasets.cifar10
# The CIFAR10 dataset contains 60,000 color images in 10 classes, 
# with 6,000 images in each class. The dataset is divided into 50,000 training
# images and 10,000 testing images.


# In[ ]:


(train_images, train_labels), (test_images, test_labels) = data.load_data()


# In[ ]:


train_images.shape


# In[ ]:


test_images.shape


# In[ ]:


print(train_labels[0])


# In[ ]:



print(train_images[1])


# In[ ]:


plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)

plt.show()


# In[ ]:


for i in range(6):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(train_images[i])
plt.show()


# In[ ]:


#Normalize

train_images, test_images = train_images / 255.0, test_images / 255.0


# In[ ]:


model = tf.keras.models.Sequential([
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(10, activation='softmax')])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))


# In[ ]:



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

