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

# coding: utf-8

# In[1]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from PIL import Image
from numpy import array
import pandas as pd
import numpy as np
import os



# In[2]:
for file in os.walk('..input/'):
    print(file)

x_train=[]
y_train=[]
def train_data():
    s_train=[]
    l_train=[]
    labels=['Images','Labels']
    rootDir='../input/'
    for file in os.walk(rootDir):
        for file_name in file[2]:
            image=Image.open(rootDir+'train/train/'+str(file_name),"r")
            image=image.resize([64,64])
            img=array(image)
            if str(file_name).startswith('cat'):
                x_train.append(img)
                y_train.append(1)
            else:
                x_train.append(img)
                y_train.append(0)
train_data()

# In[3]:
def unison_shuffled_copies(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

x_train=np.array(x_train)
y_train=np.array(y_train)
x_train,y_train = unison_shuffled_copies(x_train,y_train)
x_test=x_train[int(x_train.shape[0]*0.8):]
y_test=y_train[int(y_train.shape[0]*0.8):]
x_train=x_train[0:int(x_train.shape[0]*0.8)]
y_train=y_train[0:int(y_train.shape[0]*0.8)]


# In[4]:


print(x_train.shape)


# In[5]:


x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255


# In[6]:


num_classes = 2

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[7]:


model_2=Sequential()

model_2.add(Conv2D(32,(5,5),strides=(1,1),padding="same",input_shape=x_train.shape[1:]))
model_2.add(Activation('relu'))

model_2.add(Conv2D(32,(5,5),strides=(1,1),padding="same"))
model_2.add(Activation('relu'))

model_2.add(MaxPooling2D(pool_size=(2,2)))

model_2.add(Conv2D(32,(5,5),strides=(1,1),padding="same"))
model_2.add(Activation('relu'))

model_2.add(Conv2D(32,(5,5),strides=(1,1),padding="same"))
model_2.add(Activation('relu'))

model_2.add(MaxPooling2D(pool_size=(2,2)))
model_2.add(Dropout(0.25))

model_2.add(Flatten())

model_2.add(Dense(128))
model_2.add(Activation('relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(num_classes))
model_2.add(Activation('softmax'))

model_2.summary()


# In[ ]:


batch_size = 25

# initiate nadam optimizer
opt = keras.optimizers.nadam(lr=0.0005)

# Let's train the model using na
model_2.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_2.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_test, y_test),
              shuffle=True)


# In[ ]:

