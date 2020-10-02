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


df=pd.read_csv('../input/churn-modelling-dataset/Churn_Modelling.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe(include='all')


# In[ ]:


df.shape


# In[ ]:


x=df.iloc[:,3:13].values


# In[ ]:


print(x[0:5])


# In[ ]:


y=df.iloc[:,13].values


# In[ ]:


print(y[0:5])


# In[ ]:


# Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X_1=LabelEncoder()
x[:,1]=labelencoder_X_1.fit_transform(x[:,1])
labelencoder_X_2=LabelEncoder()
x[:,2]=labelencoder_X_1.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])

x=onehotencoder.fit_transform(x).toarray()


# In[ ]:


print(x[0])


# In[ ]:


x=x[:,1:]


# In[ ]:


# Splitting datasets into training and testing test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=365)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[ ]:


x_test


# In[ ]:


# Part 2 Building the ANN
# Importing Keras Library
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# Intializing the ANN
classifier=Sequential()


# In[ ]:


# Adding the input and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))


# In[ ]:


# Adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))


# In[ ]:


# Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


# In[ ]:


# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# Fitting the dataset into ANN
classifier.fit(x_train,y_train,batch_size=10,epochs=100)


# In[ ]:


# Part 3 Prediction of the test data
y_hat=classifier.predict(x_test)


# In[ ]:


y_hat=(y_hat>0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_hat)
print(cm)
ac=accuracy_score(y_test,y_hat)
print(ac)


# Use our ANN model to predict if the customer with the following informations will leave the bank: 
# 
# Geography: France
# 
# Credit Score: 600
# 
# Gender: Male
# 
# Age: 40 years old
# 
# Tenure: 3 years
# 
# Balance: $60000
# 
# Number of Products: 2
# 
# Does this customer have a credit card ? Yes
# 
# Is this customer an Active Member: Yes
# 
# Estimated Salary: $50000

# In[ ]:




