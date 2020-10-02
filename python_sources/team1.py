#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from imageio import imread 
from skimage.transform import resize
import os
import keras
import tensorflow


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


cd drive/'My Drive'/GoogleCollab/ConeDistanceRegression/


# In[ ]:


# load image filenames
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('sample.csv')
train_id = [str(i) for i in df_train['Id']]
test_id = [str(i) for i in df_test['Id']]


# In[ ]:


# get training images
train_images = [imread('TrainingQuarter/' + j) for j in train_id]
# resized = [resize(i, (width, height)) for i in train_images]
# train_images = np.array(resized)
train_images = np.array(train_images)


# In[ ]:


# get testing images
test_images = [imread('TestingQuarter/' + j) for j in test_id]
test_images = np.array(test_images)


# In[ ]:


# get distances and make them bigger
dist = np.array(df_train['Distance'])
dist = dist * 1000000
# print(dist)

# make a validation set
train_img = train_images[0:2700, :, :, :]
train_y = dist[0:2700]

val_img = train_images[2700:, :, :, :]
val_y = dist[2700:]

print(train_img.shape, train_y.shape)
print(val_img.shape, val_y.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers


# In[ ]:


model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(270,480,4)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.03)))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.03)))
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(1))


# In[ ]:


model = StartingModel()

model.summary()


# In[ ]:


model.fit(train_images, dist, batch_size=10, epochs=100, validation_split=0.05)


# In[ ]:


pred = model.predict(test_images)
pred = pred / 1000000


# In[ ]:


df_sol = pd.DataFrame()

df_sol['Id'] = df_test['Id']
df_sol['Distance'] = pred
df_sol.to_csv('solution_l2_5.csv', index=False)

model.save('model_l2_5.h5')

