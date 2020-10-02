#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import seaborn as sns


# In[ ]:


X_train = np.load("../input/X.npy")
Y_train = np.load("../input/Y.npy")

print("x shape: ", X_train.shape)
print("y shape: ", Y_train.shape)


# In[ ]:


#normalization
X_train = X_train / 255
X_train.shape


# In[ ]:


#reshaping data
X_train = X_train.reshape(-1,64,64,1)
X_train.shape


# In[ ]:


#train and test split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)


# In[ ]:


#create cnn model
model = Sequential()

# convolotion => max pool => Dropout 1
model.add(Conv2D(filters=64, kernel_size=(8,8), padding='Same', activation='relu', input_shape=(64,64,1)))
model.add(MaxPool2D(pool_size=(4,4)))
model.add(Dropout(0.5))

# convolotion => max pool => Dropout 2
model.add(Conv2D(filters=32, kernel_size=(4,4), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(4,4), strides=(1,1)))
model.add(Dropout(0.5))

# convolotion => max pool => Dropout 2
model.add(Conv2D(filters=16, kernel_size=(4,4), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=(1,1)))
model.add(Dropout(0.5))

#hidden layer
model.add(Flatten())
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# In[ ]:


#Define optimizer adam optimizer : changing learning rate
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)


# In[ ]:


#compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


epochs = 10
batch_size = 3


# In[ ]:


#Data Augmentation
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0.5, zoom_range=0.5, width_shift_range=0.5, height_shift_range=0.5, horizontal_flip=False, vertical_flip=False)
datagen.fit(X_train)


# In[ ]:


history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, Y_val),steps_per_epoch=X_train.shape[0] / batch_size)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




