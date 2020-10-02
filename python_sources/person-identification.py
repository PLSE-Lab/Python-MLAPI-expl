#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


#building CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


# In[ ]:


classifier = Sequential() #initialize cnn


# In[ ]:


classifier.add(Conv2D(filters=32,kernel_size=3, strides=(1,1), input_shape=(64,64,3), padding='same', activation = 'relu')) #number of feature detectors, row, column


# In[ ]:


classifier.add(MaxPooling2D(pool_size = (3,3)))


# In[ ]:


classifier.add(Conv2D(filters=64,kernel_size=3, strides=1, padding = 'same', activation = 'relu')) #number of feature detectors, row, column


# In[ ]:


classifier.add(MaxPooling2D(pool_size = (3,3)))


# In[ ]:


classifier.add(Conv2D(filters=64,kernel_size=3, strides=1, padding ='same', activation = 'relu')) #number of feature detectors, row, column


# In[ ]:


classifier.add(MaxPooling2D(pool_size = (3,3)))


# In[ ]:


classifier.add(Flatten())


# In[ ]:


classifier.add(Dense(output_dim = 512 , activation = 'tanh'))


# In[ ]:


classifier.add(Dense(output_dim = 164 , activation = 'softmax'))


# In[ ]:


classifier.compile( optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


batch_size = 32


# In[ ]:


training_set = train_datagen.flow_from_directory('../input/ear-dataset/dataset2/train',
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../input/ear-dataset/dataset2/test',
                                            target_size = (64, 64),
                                            batch_size = batch_size//2,
                                            class_mode = 'categorical')


# In[ ]:


filepath = "weightsv2.best.hdf5"


# In[ ]:


call_back = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
checkpoints = [call_back]


# In[ ]:


classifier.fit_generator(training_set,
                         steps_per_epoch = 712,
                         epochs = 30,
                         validation_data = test_set,
                         callbacks = checkpoints)


# In[ ]:




