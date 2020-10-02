#!/usr/bin/env python
# coding: utf-8

# Lets import the neccessary package required for Digit Recognition.
# 
# We will be using CNN (Convolutional Neural Network).

# In[ ]:


import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical


# Read the training set and test set.

# In[ ]:


df_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


X=df_train.drop('label',axis=1)/255
Y=to_categorical(df_train['label'])


# In[ ]:


X=np.array(X).reshape(42000,28,28,1)
X_predict=np.array(df_test/255).reshape(28000,28,28,1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)


# In[ ]:


network=Sequential()
network.add(Conv2D(128,(3,3),activation='relu',input_shape=(28,28,1)))
network.add(MaxPool2D(2,2))
network.add(Conv2D(128,(3,3),activation='relu'))
network.add(MaxPool2D(2,2))
network.add(Conv2D(128,(3,3),activation='relu'))
#network.add(MaxPool2D(2,2))
network.add(Flatten())
network.add(Dropout(0.2))
network.add(Dense(256,activation='relu'))
network.add(Dense(512,activation='relu'))
network.add(Dense(256,activation='relu'))
network.add(Dense(10,activation='softmax'))


# Above, we have used the Dropout layer to overcome the overfitting problem.

# In[ ]:


network.summary()


# In[ ]:


network.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.0001),metrics=['acc'])


# Now lets train our network with 30 epochs.

# In[ ]:


network.fit(X_train,Y_train,epochs=30,batch_size=256)


# We have completed our training.
# 
# And we got 99.50% accuracy on the training set. 
# 
# Now lets evaluate the test set to see how our model performs on the unseen data.

# In[ ]:


network.evaluate(X_test,Y_test,batch_size=128)


# Our model has performed really well. It has given 99% accuracy.
# 
# This may not be the best solution. We can adjust the dropout layer and Con2D layer to improve the model performance.
