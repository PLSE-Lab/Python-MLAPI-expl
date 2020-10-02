#!/usr/bin/env python
# coding: utf-8

# # #MNIST Digit Recognistion without CNN. 
# # # Simplest implementation with more than 97.7% testing accuracy

# # #To see CNN implementation with 99% accuracy visit
# 
# https://www.kaggle.com/rajjai3/mnist-digitrecognizer-cnn-99-accuracy

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop,SGD
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[ ]:


import os
os.listdir('../input')


# In[ ]:


train=pd.read_csv('../input/digit-recognizer/train.csv')
target_train=train['label']
data_train=train.drop('label',axis=1)/255


# In[ ]:


test=pd.read_csv('../input/digit-recognizer/test.csv')
data_test=test/255


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(data_train,target_train,test_size=0.1,random_state=47)


# In[ ]:


sea.countplot(Y_train)
plt.show()


# In[ ]:


X_train_processed=np.array(X_train).reshape(-1,28*28)
X_test_processed=np.array(X_test).reshape(-1,28*28)


# In[ ]:


Y_train_pro=to_categorical(Y_train)
Y_test_pro=to_categorical(Y_test)


# In[ ]:


network=Sequential()
network.add(Dense(512,activation='relu',input_shape=(28*28,)))
network.add(Dropout(0.005))
network.add(Dense(256,activation='relu'))
network.add(Dense(10,activation='softmax'))


# In[ ]:


network.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.09,momentum=0.9),metrics=['categorical_accuracy'])
network.fit(X_train_processed,Y_train_pro,epochs=20,batch_size=256)


# In[ ]:


network.evaluate(X_test_processed,Y_test_pro)

