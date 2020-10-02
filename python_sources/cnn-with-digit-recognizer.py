#!/usr/bin/env python
# coding: utf-8

# In[34]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from subprocess import check_output
print(check_output(['ls','../input']))


# In[ ]:





# In[35]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[36]:


print(train.shape)
train.head(5)


# In[4]:


print(test.shape)
test.head(5)


# ## Divide the training data and Label.

# In[37]:


X_train = (train.iloc[:,1:].values).astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
X_test = test.values.astype('float32')


# In[8]:


X_train.shape


# Quick Look at an Image

# In[13]:


X_train=X_train.reshape(X_train.shape[0],28,28)
plt.imshow(X_train[7],cmap=plt.get_cmap('Greys'))
plt.title(y_train[7])


# We need to do this step to standarized the data to get the data value around 0 and 1

# In[38]:


train_images=X_train.reshape((X_train.shape[0],28,28,1))
train_images=train_images.astype('float')/255

test_images=X_test.reshape((test.shape[0],28,28,1))
test_images=test_images.astype('float')/255


# ### One-Hot Encoding Of Label

# In[39]:


y_train=to_categorical(y_train)


# In[40]:


y_train.shape


# ## Split Training and Validation Set

# In[41]:


x_train,x_val,y_train,y_val=train_test_split(train_images,y_train,test_size=0.2,random_state=23)


# ## Define The Model

# In[42]:


model=Sequential()
model.add(Conv2D(32,(5,5),activation='relu',padding='same',input_shape=(28,28,1)))
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))


# In[43]:


model.summary()


# ## Set the optimizer and Compile It

# In[44]:


model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-4),
              metrics=['acc'])


# ## Augment The Data via ImageDataGenerator

# In[46]:


datagen=ImageDataGenerator(
rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
  
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

datagen.fit(x_train)


# In[48]:


data_generator=datagen.flow(x_train,y_train,batch_size=128)
history=model.fit_generator(data_generator,
                           epochs=100,
                           validation_data=(x_val,y_val))


# In[49]:


acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
ip_range=[]
for i in range(1,len(acc)+1):
    ip_range.append(i)

trace1=go.Scatter(
               x=ip_range,
               y=acc,
            mode='lines+markers',
            name='Train_Accuracy')

trace2=go.Scatter(
               x=ip_range,
               y=val_acc,
            mode='lines+markers',
            name='Validation Accuracy')

data=[trace1,trace2]
py.iplot(data)


# In[50]:


trace1=go.Scatter(
               x=ip_range,
               y=loss,
            mode='lines+markers',
            name='Train_Loss')

trace2=go.Scatter(
               x=ip_range,
               y=val_loss,
            mode='lines+markers',
            name='Validation Loss')

data=[trace1,trace2]
py.iplot(data)


# In[ ]:


predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("cnn.csv", index=False, header=True)

