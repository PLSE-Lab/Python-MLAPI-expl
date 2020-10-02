#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/seanbenhur/cifar10/blob/master/cifar10.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#invite some people to kaggle  party
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# In[ ]:


#load the data
data = data = tf.keras.datasets.cifar10
(train_images, train_labels),(test_images,test_labels) = data.load_data()


# In[ ]:


#print the shape of the dataset
print('Number of training images:',train_images.shape)
print('Number of testing images: ',test_images.shape)


# In[ ]:


#rescale the images
train_images, test_images = train_images/255.0, test_images/255.0


# In[ ]:


class_names = ['airplane','automobile','bird','cat','deer','dog',
               'frog','horse','ship','truck']


# In[ ]:


#plot some images from train dataset 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[ ]:


#create a network 
model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPooling2D((2,2)),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPooling2D((2,2)),
                             tf.keras.layers.Dropout(0.3),
                             tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPooling2D((2,2)),
                             tf.keras.layers.Dropout(0.4),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128,activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dropout(0.4),
                             tf.keras.layers.Dense(10,activation='softmax')
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(train_images, train_labels, epochs=100, batch_size=64, validation_data=(test_images,test_labels), verbose=0)


# In[ ]:


_, acc= model.evaluate(test_images, test_labels, verbose=0)
print('> %.3f' % (acc * 100.0))


# In[ ]:


#plot the accuracy 
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# In[ ]:


#predicting the test dataset
predictions = model.predict(test_images)


# In[ ]:


predictions.shape


# In[ ]:


predictions[0]


# In[ ]:


np.argmax(predictions[0])


# In[ ]:


#checking the predicted label
test_labels[0]


# In[ ]:


# Grab an image from the test dataset
img = test_images[0]

print(img.shape)


# In[ ]:


# Add the image to a batch where it's the only member.
img = np.array([img])

print(img.shape)


# In[ ]:


predictions_single = model.predict(img)

print(predictions_single)


# In[ ]:


np.argmax(predictions_single[0])


# You can see that both the predictions are same
