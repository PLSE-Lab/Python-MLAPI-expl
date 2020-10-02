#!/usr/bin/env python
# coding: utf-8

# # Basic Convolutional Neural Network
# Here is an example of a basic CNN. For images the accuracy of a CNN is better than a Densely connected Neural Network for two reasons.
# 1. CNNs are able to learn local features. A fully connected NN can only find features in the whole image. In this example we use a kernel size of 3x3 pixels. The feature window is also called the `receptive field` of a kernel. 
# 1. CNNs are able to learn a hierarchy of features. With multiple layers, feature detectors for concepts like 'face' or 'wheel' are built up from simpler detectors like 'edge' or 'curve'.
# 
# Adapted from Deep Learning with Python 5.1

# In[ ]:


from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[ ]:


model.summary()


# In[ ]:


# images are fed in as 28x28x1 tensors
# the network shrinks the height and width
# number of filters per layer determine number of channels in tensor


# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[ ]:


# we flatten the tensor and run it through a softmax layer


# In[ ]:


model.summary()


# In[ ]:


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[ ]:


model.fit(train_images, train_labels, epochs=5, batch_size=64) 


# In[ ]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[ ]:


test_acc

