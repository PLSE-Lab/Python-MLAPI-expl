#!/usr/bin/env python
# coding: utf-8

# # 0-9 Digit Recognizer using CNN-Tensorflow-Keras
# 
# * 16/01/2020

# ## Overview
# 1. Introduction
# 2. Data Preparation
# 3. Model
# 4. Evaluation
# 

# ## 1. Introduction 

# To make a good digital recognizer to recognize the digits(0-9) by training different images of digits using Convolutional Neural Networks or ConvNets. Here I used Keras API by using Tensorflow backend.

# **Importing useful packages**

# In[ ]:


import matplotlib.pyplot as plt        # for visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam,RMSprop
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from keras.utils.np_utils import to_categorical


# Retrieving datasets from directory

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


# **Data from direcory, showing shapes**

# In[ ]:


train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print('train data :',train.shape)        
print('test data :',test.shape)


# In[ ]:


train.head(3)          #train data


# In[ ]:


test.head(3)             #test data


# In[ ]:


train.label.value_counts()


# In[ ]:


sns.countplot(train.label)


# ## 2. Data Preparation

# In[ ]:


X_train=train.iloc[:,1:]              #inputs
Y_train=train.label                   #label


# In[ ]:


X_train=X_train/255                      #normalize


# In[ ]:


#reshape
X_train=X_train.values.reshape(-1,28,28,1)


# In[ ]:


X_train.shape


# In[ ]:


Y_train=to_categorical(Y_train)


# In[ ]:


Y_train.shape


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.1,random_state=42)


# In[ ]:


plt.imshow(x_train[30000][:,:,0])


# ## 3. Modelling

# In[ ]:


model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(BatchNormalization(momentum=0.15))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(BatchNormalization(momentum=0.15))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(BatchNormalization(momentum=0.15))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


#optimizer
optimzr=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)


# In[ ]:


model.compile(optimizer=optimzr,loss=['categorical_crossentropy'],metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train,batch_size=20,epochs=5)


# ## 4. Evaluation

# ### On train set

# In[ ]:


y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
y_test=np.argmax(y_test,axis=1)
print('accuracy :',accuracy_score(y_test,y_pred))
print('confusion_matrix :',pd.DataFrame(confusion_matrix(y_test,y_pred),index=range(0,10),columns=range(0,10)))
print('classification_report :',classification_report(y_test,y_pred))


# ## on test set

# **To predict for our test set
# so whatever we did in training set we need to do the same**

# In[ ]:


test=test/255           #normalize
test=test.values.reshape(-1,28,28,1)        #reshape
print('test shape:',test.shape)

y_predict=model.predict(test)
print(y_predict)
y_predict=np.argmax(y_predict,axis=1)
print(y_predict)




# In[ ]:


y_predict.shape


# ### to upload in the same format of submission

# In[ ]:


my_submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')


# In[ ]:


my_submission.head(2)


# In[ ]:


my_submission['Label']=y_predict


# In[ ]:


my_submission.to_csv('my_labels',index=False)


# In[ ]:




