#!/usr/bin/env python
# coding: utf-8

# In[135]:


import numpy as np 
import pandas as pd 
dataset = pd.read_csv('../input/Churn_Modelling.csv')
dataset.head()


# In[136]:


dataset.info()


# There are no null values in the data, hence the data is clean. Remove the columns that are not required for the dataset and changing the dataframe to numpy array by dividing the dependent and independent variables.

# In[137]:


dataset = dataset.drop(['RowNumber','CustomerId','Surname'], axis=1)
dataset.head()


# Feature selection, converting the categorical to numerical.

# In[138]:


geography = pd.get_dummies(dataset['Geography'],drop_first=True)
# similarly for this colimn as well. If there are n dummy columns, consider n-1
gender = pd.get_dummies(dataset['Gender'],drop_first=True)


# In[139]:


dataset = dataset.drop(['Geography','Gender'], axis=1)
dataset = pd.concat([dataset,geography,gender],axis=1)
dataset.head()


# Spearating dependent and independent variables.

# In[140]:


X = dataset.drop("Exited",axis=1)
y = dataset['Exited']


# In[141]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[142]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[143]:


#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# Initialising the ANN
classifier = Sequential()


# In[144]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))


# In[145]:


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[146]:


classifier.summary()


# In[147]:


#Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)


# In[148]:


#history
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred[:5]


# In[149]:


# Making the classification report
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))


# In[150]:


print(accuracy_score(y_test, y_pred)*100)


# In[151]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[152]:


score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])


# In[153]:


# change the epochs to 5, 10 from 2
# we got 79% acc with 2 & 5 & 20 epochs with SGD
# we got 83% acc with 20 epochs with adam
# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)


# In[154]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred[:5]


# In[155]:



from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))


# Making the Confusion Matrix

# In[156]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[159]:


score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])


# In[160]:


print(accuracy_score(y_test, y_pred)*100)

