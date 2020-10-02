#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df= pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


df.shape, df.head(), df.tail()


# In[ ]:


X= df.iloc[:,1:].values
y= df.iloc[:,0].values


# In[ ]:


X.shape,X


# In[ ]:


y.shape,y, np.unique(y)


# * Here I performed the train_test_split without using any external libraries

# In[ ]:


train_length = len(X)*0.75
train_length


# In[ ]:


train_X,test_X = X[:int(train_length),:],X[int(train_length):,:]
train_y,test_y=y[:int(train_length)], y[int(train_length):]


# In[ ]:


train_X.shape, test_X.shape, train_y.shape, test_y.shape


# * Reshaping the data into required format so that it can trained in a neural net.

# In[ ]:


train_X = train_X.reshape(train_X.shape[0],28,28,1)
test_X = test_X.reshape(test_X.shape[0],28,28,1)

train_X.shape, test_X.shape


# * Normalizing the data

# In[ ]:


train_X= train_X/255
test_X= test_X/255


# * One hot encoding to the dependent variable
# 

# In[ ]:


import keras 

train_y= keras.utils.to_categorical(train_y,10)
test_y= keras.utils.to_categorical(test_y,10)


# # CNN Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


# In[ ]:


model = Sequential()

model.add(Conv2D(64,(3,3), activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))


model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch,logs={}):
    if(logs.get('loss')<0.05):
      print('Accuracy is more than 99%')
      self.model.stop_training=True

callbacks=myCallback()


# In[ ]:


model.fit(train_X, train_y, batch_size=16, epochs= 50, verbose=1, validation_data=(test_X,test_y),callbacks=[callbacks])


# In[ ]:


score= model.evaluate(test_X,test_y)
print('loss',score[0])
print('accuracy',score[1])


# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


test.head()


# In[ ]:


test= test.values.reshape(test.shape[0],28,28,1)


# In[ ]:


test.shape


# * predicting for the test data

# In[ ]:


predicted=model.predict_classes(test)


# In[ ]:


predicted[0]


# In[ ]:


sub=pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[ ]:


sub.head()


# In[ ]:


sub['Label']= predicted


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('../input/output/submission.csv',index=False)


# In[ ]:




