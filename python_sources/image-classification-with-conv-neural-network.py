#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load the extension and start TensorBoard
"""!kill 30
%load_ext tensorboard
%tensorboard --logdir logs"""


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
            os.path.join(dirname, filename)      

# Any results you write to the current directory are saved as output.


# # Import Image Data and Convert to Features and Labels

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2

DATADIR='/kaggle/input/cats-and-dogs-sentdex-tutorial/kagglecatsanddogs_3367a/PetImages'
CATEGORIES=['Cat','Dog']

for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break


# In[ ]:


print(img_array.shape)


# In[ ]:


IMG_SIZE=50
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[ ]:


training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()            


# In[ ]:


print(len(training_data))


# In[ ]:


import random
random.shuffle(training_data)


# In[ ]:


for sample in training_data:
    print(sample[1])


# In[ ]:


X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


print(X.shape)


# # Building a Simple CNN

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten


# In[ ]:


X=X/255.0


# In[ ]:


print(X.shape)


# In[ ]:


model=Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# # Using TensorBoard to Plot the Curve

# In[ ]:


from tensorflow.keras.callbacks import TensorBoard
import datetime
import time
#Name="cats-vs-dogs-cnn-{}".format(int(time.time()))
#tensorboard=TensorBoard(log_dir='logs/{}'.format(Name))
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[ ]:


y=np.array(y)
model.fit(X,y, batch_size=32, epochs=1, validation_split=0.1)

