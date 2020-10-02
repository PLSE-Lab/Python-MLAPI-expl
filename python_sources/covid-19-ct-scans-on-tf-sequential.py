#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
np.random.seed(123)
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,                          Flatten, Convolution2D, MaxPooling2D,                          BatchNormalization, UpSampling2D
from keras.utils import np_utils
import tensorflow as tf

from skimage.io import imread
from sklearn.model_selection import train_test_split


# In[ ]:


# checking whether we have loaded the datasets we need and their input names
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# set channels first notation
K.common.set_image_dim_ordering('th')


# In[ ]:


jimread = lambda x:np.expand_dims(imread(x)[::4, ::4], 0)


# In[ ]:


# Dataset: Covid-19 Patients Lung X ray images 10000
data = "../input/covid-19-x-ray-10000-images/dataset"


# In[ ]:


os.listdir(data)


# In[ ]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

normal_images = []
for img_path in glob.glob(data + '/normal/*'):
    normal_images.append(mpimg.imread(img_path))

fig = plt.figure()
fig.suptitle('normal')
plt.imshow(normal_images[0], cmap='gray') 

covid_images = []
for img_path in glob.glob(data + '/covid/*'):
    covid_images.append(mpimg.imread(img_path))

fig = plt.figure()
fig.suptitle('covid')
plt.imshow(covid_images[0], cmap='gray')


# In[ ]:


w = 150
h = 150 
channels = 3

input_shape = (w, h, channels)
nb_classes = 2
epochs = 25
batch_size = 6


# In[ ]:


model = Sequential()
model.add(Convolution2D(filters = 32,
                       kernel_size = (3, 3),
                       activation = "relu",
                       input_shape = input_shape,
                       padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filters = 32,
                       kernel_size = (3, 3),
                       activation = "relu", 
                       input_shape = input_shape, 
                       padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filters = 64, 
                       kernel_size = (3, 3),
                       activation = "relu",
                       input_shape = input_shape,
                       padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Convolution2D(filters = 128,
                       kernel_size = (3, 3),
                       activation = "relu", 
                       input_shape = input_shape, 
                       padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filters = 256, 
                       kernel_size = (2, 2),
                       activation = "relu", 
                       input_shape = input_shape, 
                       padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.50))
model.add(Dense(1))
model.add(Activation("sigmoid"))


# In[ ]:


optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001)


# In[ ]:


model.compile(loss = "binary_crossentropy",
             optimizer = optimizer, 
             metrics = ['accuracy'])
loss_history = []
print(model.summary())


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    data,
    target_size=(h, w),
    batch_size= batch_size,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data, 
    target_size=(h, w),
    batch_size= batch_size,
    class_mode='binary',
    shuffle= False,
    subset='validation')

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = epochs)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


pred= model.predict(validation_generator)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (validation_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print (labels)
print (predictions)


# In[ ]:


label = validation_generator.classes


# In[ ]:


from sklearn.metrics import confusion_matrix

cf = confusion_matrix(predicted_class_indices,label)
cf


# In[ ]:


import seaborn as sns
fig = plt.figure(figsize = (12, 9))
sns.heatmap(cf, annot = False, cmap = "YlGnBu")


# In[ ]:




