#!/usr/bin/env python
# coding: utf-8

# # Starting with ANN

# ***HOLA AMIGOS!!!***
# 
# Do upvote if you find the the NoteBook helpful.

# **Artificial neural networks (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains.** 

# this is how a basic model looks like

# ![1_Gh5PS4R_A5drl5ebd_gNrg@2x.png](attachment:1_Gh5PS4R_A5drl5ebd_gNrg@2x.png)

# Today, we will try to predict the probablity of the customer to buy a certain product on the basis of the given paprameters that are  ***'GENDER'***, ***'AGE'***, ***'ESTIMATED SALARY'***.
# 

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


# # 1. Importing libraries

# **Import all the necessary libraries.**

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder


# In[ ]:


print(tf.__version__)


# # 2. Loading the Dataset

# importing the dataset (CSV) file.

# In[ ]:


dataset = pd.read_csv('/kaggle/input/Social_Network_Ads.csv')


# In[ ]:


dataset.head() 


# # 3. Processing the Data

# *  checking if there are any null values in DataSet.

# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.drop(columns=['User ID'],inplace = True)


# * slicing the data in the train and test set respectively.

# In[ ]:


x = dataset.iloc[:,:3]
y = dataset.iloc[:,3]


# * Applying LabelEncoder to the column *Gender* so that the categorical values can be taken as numbers when trained

# In[ ]:


le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
x


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 5)


# In[ ]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# # 4. Building the ANN Model layers

# # Activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The motive is to introduce non-linearity into the output of a neuron.

# Here, we have tried to make the architecture of the model by calling the Sequential library from Keras

# 1. Dense
# 2. Flatten 
# 3. Activation
# 

# In[ ]:


model = Sequential()

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))


# * we have used the *ADAM* Gradient Descent here, with the loss as *Binary Crossentropy*

# In[ ]:


model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])


# # 5. Training the model

# In[ ]:


model.fit(x_train,y_train ,batch_size = 16 ,epochs = 100)


# # 6. Predictions and Evaluating of our the model

# In[ ]:


y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


# In[ ]:


print('percentage Accuracy : ',100*accuracy_score(y_test,y_pred))


# 

# ### Predicting the result of a single observation

# where the customer is ***MALE***, with age ***36*** and salary ***33000***

# In[ ]:


pred = model.predict(sc.transform([[1, 36, 33000]])) > 0.5
if pred == True:
    print('1 : True')
else:
    print('0 : False')


# In[ ]:




