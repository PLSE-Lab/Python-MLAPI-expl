#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the Keras libraries and packages
import numpy as np


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[ ]:


test_dir="../input/dogs-cats-images/dog vs cat/dataset/test_set"
train_dir="../input/dogs-cats-images/dog vs cat/dataset/training_set"


# In[ ]:





# In[ ]:


# Initializing the CNN
classifier = Sequential()


# In[ ]:


# Step 1 - Convolution
classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = "relu"))


# In[ ]:


# Step 2 - Maxpooling
classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[ ]:


# Step 3 - Flattening
classifier.add(Flatten())


# In[ ]:


# Step 4 - Full Connection to the ANN
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[ ]:


# Step 5 - Compiling 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.summary()


# In[ ]:


# Step 6 - Fitting CNN to the training set
# From Keras Image Processing Website
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32,class_mode='binary')


# In[ ]:


classifier.fit_generator(training_set, steps_per_epoch=2000, epochs=5, validation_data = test_set, validation_steps=60)


# In[ ]:


res_list= ["It's a cat !","It's a dog !"]


# In[ ]:


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
test_image = image.load_img("../input/dogs-cats-images/dog vs cat/dataset/training_set/dogs/dog.1056.jpg", target_size = (64, 64)) 
plt.imshow(test_image)
plt.grid(None) 
plt.show()


# In[ ]:


test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


# In[ ]:


print(res_list[int(classifier.predict(test_image))])


# In[ ]:


classifier.predict(test_image)


# In[ ]:




