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


import matplotlib.pyplot as plt
data=pd.read_csv('/kaggle/input/churn-modellingcsv/Churn_Modelling.csv')


# In[ ]:


X=data.iloc[:,3:13]
y=data.iloc[:,13]


# wrangling the data. getting geography and gender in numerical varibale

# In[ ]:


geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


# In[ ]:


X=pd.concat([X,geography,gender],axis=1)


# In[ ]:


X=X.drop(['Geography','Gender'],axis=1)


# In[ ]:


X.head()


# *splitting the data*

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **using standard scaler to scaler down the data as in neural network distance is involved that is why we have to scale our values**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **let make an ANN for our data **

# importing required libaries

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[ ]:


classifier=Sequential()
#adding first input hidden layer of neurons
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))
#adding 2nd layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
#adding 3rd layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))


# **this is what our neural net look like**

# In[ ]:


classifier.summary()


# In[ ]:


# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 20, nb_epoch = 100)


# In[ ]:


print(model_history.history.keys())


# In[ ]:


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# part 3 of our model in this we will be making predictions

# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# In[ ]:


cm


# In[ ]:


score


# # **if you like the kernel plaese upvote it**

# In[ ]:




