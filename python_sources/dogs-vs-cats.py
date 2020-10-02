#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#BUILDING THE cnn
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
print(os.listdir("../input"))


# In[ ]:


#intialising the CNN
classifier=Sequential()


# In[ ]:


#step-1 Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))


# In[ ]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[ ]:


#second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[ ]:


classifier.add(Flatten())


# In[ ]:


classifier.add(Dense(output_dim= 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


#image augmentation
train_datagen= ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[ ]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


training_set = train_datagen.flow_from_directory('../input/dataset/dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[ ]:


test_set = test_datagen.flow_from_directory('../input/dataset/dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[ ]:


classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)


# In[ ]:




