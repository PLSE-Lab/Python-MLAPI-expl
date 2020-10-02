#!/usr/bin/env python
# coding: utf-8

# <h3><font color='#50474F'> Hello and welcome to this notebook where YOU are going to discover how to build CNN model :))</font></h3>
# <h1><font color='#50474F'> 1] Loading Modules that we're going to need </font></h1>

# In[ ]:


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, add, Flatten, MaxPooling2D, Conv2D, Dense, AveragePooling2D


# <h1><font color='#50474F'> 2] Loading & Spliting the Data </font></h1>

# In[ ]:


#Loading the Data
df = pd.read_csv("../input/digit-recognizer/train.csv")
X = np.c_[df.iloc[:, 1:]].reshape(len(df), 28, 28, 1)
y = np.c_[df.iloc[:, 0]]

#Transforming the labels to OneHotVectors (e.g, Label = 2 Becomes after OneHotting => [0, 0, 1, 0, 0, 0, 0 ,0, 0, 0] ,Label = 0 Becomes after OneHotting => [1, 0, 0, 0, 0, 0, 0 ,0, 0, 0] )
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
enc_label = OneHotEncoder()
y = enc_label.fit_transform(y).toarray()
#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15)


# <h1><font color='#50474F'> 3] Building Our Model! </font></h1>

# In[ ]:



model = Sequential()
#INPUT LAYER
model.add(Conv2D(32, (4, 4), input_shape=(28, 28, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2)))
#HIDDEN LAYER 1
model.add(Conv2D(32, (4, 4), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
#HIDDEN LAYER 2
model.add(Conv2D(32, (4, 4)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D((4, 4)))
#HIDDEN LAYER 3
model.add(Conv2D(32, (2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
#FULLY CNNECTED LAYER
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
model.add(Activation('relu'))
#OUTPUT LAYER
model.add(Dense(10))
model.add(Activation('softmax'))
#CHOOSING our MODEL's loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# <h1><font color='#50474F'> 3] Building The Flow of Images THEN Training our Model </font></h1>

# In[ ]:


#Building the Flow of Images
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator()
train_gen = train_gen.flow(X, y, batch_size=1000)


# In[ ]:


#Training the model
model.fit_generator(train_gen, epochs=10, steps_per_epoch=10000)


# <h1><font color='#50474F'> Let's see how well our model Generalizes :D </font></h1>

# In[ ]:


loss, acc = model.evaluate(X_test, y_test)
print('Loss : {}, Accuracy : {}'.format(loss, acc))


# <h3><font color='#50474F'>Good Enough!, but i would suggest training this model on the whole given dataset and this well produce a better result! :)), See you in the next notebook!<font></h3>

# <h1><center><font color='#F21212'>An investement in knowledge pays the best interest :))</font></center></h1>

# In[ ]:




