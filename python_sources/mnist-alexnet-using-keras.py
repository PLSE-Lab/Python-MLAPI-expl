#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('pylab', 'inline')

import copy
import matplotlib.pyplot as plt

from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical

from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.shape)
train.head()


# In[6]:


test= pd.read_csv("../input/digit-recognizer/test.csv")
print(test.shape)
test.head()


# In[7]:


X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')


# In[8]:


X_train


# In[9]:


y_train


# In[10]:


#Convert train datset to (num_images, img_rows, img_cols) format 
X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(0,9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);


# In[11]:


#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape


# In[12]:


X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape


# In[13]:


mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px


# In[14]:


y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes


# In[15]:


plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10));


# # Lenet-5

# In[30]:


model = Sequential()

model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("sigmoid"))

model.add(Convolution2D(16, 5, 5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("sigmoid"))
model.add(Dropout(0.5))

model.add(Convolution2D(120, 1, 1, border_mode='valid'))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[31]:


model.summary()


# In[32]:


model.compile(optimizer=SGD(lr=0.8),loss='categorical_crossentropy',metrics=['accuracy'])


# In[33]:


from keras.preprocessing import image
gen = image.ImageDataGenerator()


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)


# In[35]:


history=model.fit_generator(batches, batches.n, nb_epoch=2,validation_data=val_batches, nb_val_samples=val_batches.n, verbose=1)

