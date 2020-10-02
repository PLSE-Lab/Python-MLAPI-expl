#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[ ]:


classifier = Sequential()


# In[ ]:


classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = 'relu'))


# In[ ]:


classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[ ]:


classifier.add(Flatten())


# In[ ]:


classifier.add(Dense(units = 128, activation = 'relu'))


# In[ ]:


classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator( rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)


# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


training_set = train_datagen.flow_from_directory("../input/train_1/train", target_size = (64,64), batch_size = 32, class_mode = 'binary')


# In[ ]:


test_set = test_datagen.flow_from_directory("../input/test_1/test", target_size = (64,64), batch_size = 32, class_mode = 'binary')


# In[ ]:


classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 2, validation_data = test_set, validation_steps = 2000)


# In[ ]:


training_set.class_indices


# In[ ]:


from keras.preprocessing import image
test_image = image.load_img('../input/pred_1/pred/pred/cat.jpg', target_size = (64,64))


# In[ ]:


test_image = image.img_to_array(test_image)


# In[ ]:


test_image = np.expand_dims(test_image,axis = 0)


# In[ ]:


result = classifier.predict(test_image)

