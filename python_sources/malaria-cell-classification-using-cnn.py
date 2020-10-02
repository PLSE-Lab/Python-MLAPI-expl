#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu'))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2,2), strides = 2))


# In[ ]:


model.add(Flatten())


# In[ ]:


model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))


# In[ ]:


model.add(Dense(units = 1, activation = 'sigmoid'))


# In[ ]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
horizontal_flip=True)


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(64, 64),
        batch_size=32,
class_mode='binary')


# In[ ]:


from PIL import Image
import keras
import os


# In[ ]:


os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"


# In[ ]:


model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)


# In[ ]:


model.save("malaria.h5")


# In[ ]:


model.load_weights('malaria.h5')


# In[ ]:


import numpy as np
from keras.preprocessing import image


# In[ ]:


path = input("Enter Image Path: ")
img = image.load_img(path=path, target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
result = model.predict(img)
if result[0][0] ==0:
    print('Parasitic')
else:
    print('Uninfected')


# In[ ]:




