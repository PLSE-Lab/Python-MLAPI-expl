#!/usr/bin/env python
# coding: utf-8

# *Import libraries (tensorflow backend)*

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


# *Build the CNN*

# In[ ]:


classifier = Sequential()


# ***Convolution***

# In[ ]:


classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))


# Choose 32 feature detectors and an input shape of 3D image of 64x64 pixels

# ***Pooling***

# In[ ]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Pooling is made with a 2x2 array 

# Add 2nd convolutional layer with the same structure as the 1st to improve predictions

# In[ ]:


classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# ***Flattening***

# In[ ]:


classifier.add(Flatten())


# ***Full Connection***

# In[ ]:


classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 1))


# CNN has 128 nodes in the first layer of the ANN that's connected in the backbone with rectifier activation function.  We then add sigmoid activation function because we have binary outcome with 1 node in the output layer.

# In[ ]:





# ***Compile the CNN***

# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# adam is for stohastic gradient descent and binary crossentropy for logarithmic loss for binary outcomes

# ***Image Augmentation***

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# Apply several transformations to train the model in a better significant way, keras documentation provides all the required information for augmentation

# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# Target size is 64x64. This is the size of the images the model is trained above and size of the batches in which random samples of our images will be included. Class mode is binary because dependent variable is binary.

# classifier.summary()

# In[ ]:


history = classifier.fit_generator(training_set,
                         steps_per_epoch = 163,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 624)


# 

# In[ ]:


#Accuracy
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()

#Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()






# 

# 
