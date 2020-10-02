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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D


# In[ ]:


train = pd.read_csv('../input/Kannada-MNIST/train.csv')
train.head()


# In[ ]:


y_train = np.array(train['label'])
y_train


# In[ ]:


X_train = train.drop(['label'],1)
X_train.head()


# In[ ]:


X_train = np.array(X_train).reshape(X_train.shape[0],28,28)
X_train.shape


# In[ ]:


plt.imshow(X_train[0])


# In[ ]:


from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10)
print(y_train[:10])


# In[ ]:


X_train = np.array(X_train/255)

from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(X_train, y_train, test_size=0.2)

x_train=x_train.reshape(x_train.shape[0], 28,28,1)
x_valid=x_valid.reshape(x_valid.shape[0],28,28,1)



# define the model

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[ ]:


from keras.preprocessing. image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

checkpoint=ModelCheckpoint('bestweights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode="max")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

datagenerator=ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1)

datagenerator.fit(x_train)
batch_size=64

epochs = 30

bm=model.fit_generator(datagenerator.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,validation_data=(x_valid, y_valid), verbose=2,steps_per_epoch=x_train.shape[0]//batch_size, callbacks=[checkpoint])
model.load_weights("bestweights.hdf5")


# In[ ]:


test = pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


test = test.drop(['id'],1)


# In[ ]:


X_test = np.array(test/255)

X_test = X_test.reshape(X_test.shape[0],28,28)

X_test=X_test.reshape(X_test.shape[0],28,28,1)

X_test.shape


# In[ ]:





# In[ ]:


results=model.predict(X_test)


# In[ ]:


results=np.argmax(results, axis=1)


# In[ ]:


results


# In[ ]:


cnn3 = pd.DataFrame({'label':results}).reset_index().rename(columns = {'index':'id'})
cnn3.head()


# In[ ]:


cnn3.to_csv('cnn3_img_aug.csv',index = False)


# In[ ]:




