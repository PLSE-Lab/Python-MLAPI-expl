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


# ## Importing Libraries

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten                                      
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


import tensorflow as tf
tf.__version__


# In[ ]:


train_df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# ## Dividing the values into x and y

# In[ ]:


x=train_df.drop('label',axis=1)
x


# In[ ]:


x.shape


# In[ ]:


y=train_df['label']
y


# ## Reshaping and converting into images

# In[ ]:


x=x.values.reshape(-1,28,28,1)


# In[ ]:


x.shape


# ## Ploting the images

# In[ ]:


plt.imshow(x[100][:,:,0])


# In[ ]:


y[100]


# ## Diving the dataframe into train and test data set

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)


# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Normalization

# In[ ]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# ## Building The CNN Layers

# In[ ]:


model=Sequential()


# In[ ]:


model.add(Convolution2D(32,(5,5),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(Convolution2D(64,(5,5),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Dropout(0.2))
model.add(Convolution2D(128,(3,3),activation='relu',padding='same'))
model.add(Convolution2D(192,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


filepath = os.path.join("./model_v{epoch}.hdf5")

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
callbacks = [checkpoint]


# In[ ]:


model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_test,y_test),callbacks=callbacks)


# ## Testing The model

# In[ ]:


model.evaluate(x_test,y_test)


# ### Model is 99.310% Accurate.

# ## Predicting on test data set

# In[ ]:


test_df=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


test_df=test_df.values.reshape(-1,28,28,1)


# In[ ]:


test_df.shape


# In[ ]:


test_df=test_df/255


# In[ ]:


classifier=load_model('./model_v14.hdf5')


# In[ ]:


prediction=classifier.predict(test_df)


# In[ ]:


prediction.shape


# In[ ]:


pr=classifier.predict_classes(test_df)


# In[ ]:


sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sub['Label'] = pr
sub.to_csv('submission1.csv',index=False)

