#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# ### Load the datasets 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ### Let's look at the shape and few example records

# In[ ]:


print('Train contains %d rows and %d columns'%(train.shape[0],train.shape[1]))
print('Test contains %d rows and %d columns'%(test.shape[0],test.shape[1]))


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Seperate the label column 

# In[ ]:


train_data = train.drop(['label'],axis=1)
train_label = train[['label']]


# ### From the columns, it looks like the images are 28*28 pixels. Lets convert the data to numpy array and reshape the data

# In[ ]:


train_img = np.array(train_data).reshape(-1,28,28,1)
test_img = np.array(test).reshape(-1,28,28,1)


# In[ ]:


print(train_img.shape,test_img.shape)


# ### Let's print few images and check 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig, ax = plt.subplots(3,3,figsize = (8,8))
for i, j in enumerate(ax.flat):
    j.imshow(train_img[i][:,:,0], cmap ='binary')
    j.text(2,2,str(train_label.label[i]),color = 'red')


# ### Data Augmentation

# In[ ]:


from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization


# ### Training a CNN model

# In[ ]:


batch_size = 32
np_classes = 10
nb_epoch = 10
#Normalize the data
x_train = train_img.astype('float')
x_test = test_img.astype('float')
x_train /= 255
x_test /= 255


# ### Convert the label to categorical

# In[ ]:


from keras.utils import to_categorical
y_train = to_categorical(train_label, np_classes)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D, MaxPool2D


# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(3,3), padding='same', input_shape = (28,28,1),activation='relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(2,2), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters=64, kernel_size=(2,2), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(np_classes, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train,batch_size = batch_size, epochs = nb_epoch, shuffle = True)


# In[ ]:


pred = model.predict_classes(x_test).reshape(-1).astype(np.int8)


# In[ ]:


submis = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submis['Label'] = pred


# In[ ]:


submis.to_csv('digit_recog_v1.csv', index=False)


# In[ ]:




