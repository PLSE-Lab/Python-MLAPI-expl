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


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


my_data_dir = '/kaggle/input/fruit-recognition/'


# In[ ]:


os.listdir(my_data_dir)


# In[ ]:





# In[ ]:


apple = my_data_dir+'Apple/'+'Total Number of Apples'
peach = my_data_dir+'Peach'
carambola = my_data_dir+'Carambola'
kiwi = my_data_dir+'Kiwi/'+'Total Number of Kiwi fruit'
tomatoes = my_data_dir+'Tomatoes'
persimmon = my_data_dir+'Persimmon'
plum = my_data_dir+'Plum'
guava = my_data_dir+'Guava/'+'guava total final'
pear = my_data_dir+'Pear'
mango = my_data_dir+'Mango'
muskmelon = my_data_dir+'muskmelon'
banana = my_data_dir+'Banana'
pomegranate = my_data_dir+'Pomegranate'
pitaya = my_data_dir+'Pitaya'
orange = my_data_dir+'Orange'


# In[ ]:


A_img= apple+'/'+'Apple 03441.png'


# In[ ]:


A_img


# In[ ]:


os.listdir(apple)


# In[ ]:


an_A_img = imread(A_img)


# In[ ]:


plt.imshow(an_A_img)


# In[ ]:


os.listdir(apple)


# In[ ]:


dim1 = []
dim2 = []
for image_filename in os.listdir(apple):
    
    img = imread(apple+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[ ]:


sns.jointplot(dim1,dim2)


# In[ ]:


np.mean(dim1)


# In[ ]:


np.mean(dim2)


# In[ ]:


image_shape = (283,383,3)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# help(ImageDataGenerator)


# In[ ]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[ ]:


image_gen.flow_from_directory(my_data_dir)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D


# In[ ]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(256))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

model.add(Dense(15))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stop = EarlyStopping(patience=2)


# In[ ]:


batch_size = 64


# In[ ]:


train_image_gen = image_gen.flow_from_directory(my_data_dir,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical')


# In[ ]:


train_image_gen.class_indices


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


results = model.fit_generator(train_image_gen,epochs=2,callbacks=[early_stop])


# ## WOW! 91 percent accuracy :O

# In[ ]:


from tensorflow.keras.models import load_model
model.save('fruit.h5')


# In[ ]:


metrics = pd.DataFrame(model.history.history)


# In[ ]:


metrics.plot()


# In[ ]:


from tensorflow.keras.preprocessing import image


# In[ ]:


#pred_probabilities = model.predict_generator(train_image_gen)


# ## Lets Predict on an image

# # APPLE

# In[ ]:


apple = apple+'/'+os.listdir(apple)[1]


# In[ ]:


my_image = image.load_img(apple,target_size=image_shape)


# In[ ]:


my_image


# In[ ]:


type(my_image)


# In[ ]:


my_image = image.img_to_array(my_image)


# In[ ]:


type(my_image)


# In[ ]:


my_image.shape


# In[ ]:


my_image = np.expand_dims(my_image, axis=0)


# In[ ]:


my_image.shape


# In[ ]:


model.predict(my_image)


# ## Recall 0 is the index for apple so that is a perfect job

# In[ ]:


train_image_gen.class_indices

