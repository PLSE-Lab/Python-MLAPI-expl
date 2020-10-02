#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam


# In[ ]:


train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
test= pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')


# In[ ]:


train.shape, train.head(), test.shape, test.head()


# In[ ]:


X = train.iloc[:,1:].values
y = train.iloc[:,0].values


# In[ ]:


X.shape,X,y.shape,y


# In[ ]:


X=X/255


# In[ ]:


train_X, test_X = X[:int(len(X)*0.75),:],X[int(len(X)*0.75):,:]
train_y, test_y =y[:int(len(X)*0.75)],y[int(len(X)*0.75):]


# In[ ]:


train_X.shape, test_X.shape, train_y.shape, test_y.shape


# In[ ]:


train_X= train_X.reshape(train_X.shape[0],28,28,1)
test_X= test_X.reshape(test_X.shape[0],28,28,1)


# In[ ]:


np.unique(y)


# In[ ]:


train_y= keras.utils.to_categorical(train_y,num_classes=25)
test_y= keras.utils.to_categorical(test_y,num_classes=25)


# In[ ]:


model = Sequential()

model.add(Conv2D(64,(3,3),activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(84,activation='relu'))

model.add(Dense(25,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


class myCallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.05):
            print('accuracy is more then 99%')
            self.model.stop_training=True
            
callbacks = myCallbacks()


# In[ ]:


model.fit(train_X, train_y, epochs=50,batch_size=32, verbose=1, validation_data=(test_X,test_y),callbacks=[callbacks])


# In[ ]:


validate_X=test.iloc[:,1:].values


# In[ ]:


validate_X=validate_X/255


# In[ ]:


validate_X= validate_X.reshape(validate_X.shape[0],28,28,1)


# In[ ]:


predicted = model.predict_classes(validate_X)


# In[ ]:


true_y= test.iloc[:,0].values


# In[ ]:


from sklearn.metrics import accuracy_score

acc = accuracy_score(true_y, predicted)
print(acc)


# In[ ]:


score = model.evaluate(test_X,test_y)
score[1]


# In[ ]:




