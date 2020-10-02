#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification with Keras
# - In this notebook, I will try to train a neural network to predict whether or not the passengers of the titanic survived the sinking given their passenger features. This notebook is in no way intended to be a competitive solution to the competiton itself but rather my personal exploration as a new user of Keras.

# In[ ]:


# Importing Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


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


# Reading in data and dropping unecessary columns
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data.head()


# In[ ]:


# Dropping missing values and viewing descriptive statistics of data 
data=data.dropna()
data.describe()


# In[ ]:


# Encoding categorical variables and standardizing continuous variables
data['Sex'] = pd.get_dummies(data['Sex'],drop_first=True)
data['Embarked'] = pd.get_dummies(data['Embarked'],drop_first=True)

data['Age'] = (data['Age'] - data['Age'].mean() ) / data['Age'].std()
data['Fare'] = (data['Fare'] - data['Fare'].mean() ) / data['Fare'].std()

data.describe()


# In[ ]:


data['Embarked']


# In[ ]:


X,y = data.drop("Survived",axis=1).values,data['Survived'].values


# In[ ]:


# Breaking data into train/test split for simple model evaluation
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=101)


# In[ ]:


# Building a neural network
from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(25,input_shape = (X_train.shape[1],),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(75,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(75,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')
model.summary()


# In[ ]:


# Training 
model.fit(X_train,y_train,epochs = 500)


# In[ ]:


# Results
from sklearn.metrics import classification_report,confusion_matrix
preds = model.predict(X_test)
preds = [i > .50 for i in preds]

print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))


# In[ ]:


# Code adapted from DataCamp.com
# Creates a model given an activation and learning rate
from keras.optimizers import Adam

def create_model(learning_rate=0.01, activation='relu'):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr=learning_rate)

    # Create your binary classification model  
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# # BE CAREFUL RUNNING THE NEXT CELL

# In[ ]:


# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV,KFold

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation':['relu', 'tanh'],  
          'epochs':[50, 100, 200], 'learning_rate':[0.1, 0.01, 0.001]}

# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3))

# random_search.fit(X,y) takes too long! But would start the search.
random_search.fit(X,y)


# In[ ]:


print("Best Params: {}".format(random_search.best_params_))
print("Best Score: {}".format(random_search.best_score_))


# In[ ]:


# Creates a model given an activation and learning rate
from keras.optimizers import Adam

def create_model(learning_rate=0.001, activation='tanh'):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr=learning_rate)

    # Create your binary classification model  
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


# Import KerasClassifier from keras wrappers
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv=3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())


# In[ ]:


# Final model trained on the entire data set

final_model = Sequential()
final_model.add(Dense(128, input_shape=(X.shape[1],),activation='tanh'))
final_model.add(Dense(256,activation='tanh'))
final_model.add(Dense(1,activation='sigmoid'))

final_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
final_model.summary()


# In[ ]:


final_model.fit(X,y,epochs=100)


# In[ ]:


# Reading in test data and dropping unecessary columns
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data = test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test_data.head()


# In[ ]:


# Encoding categorical variables and standardizing continuous variables
test_data['Sex'] = pd.get_dummies(test_data['Sex'],drop_first=True)
test_data['Embarked'] = pd.get_dummies(test_data['Embarked'],drop_first=True)

test_data['Age'] = (test_data['Age'] - test_data['Age'].mean() ) / test_data['Age'].std()
test_data['Fare'] = (test_data['Fare'] - test_data['Fare'].mean() ) / test_data['Fare'].std()

test_data.describe()


# In[ ]:


# Predicting on test data
test_data = test_data.values
preds = final_model.predict(test_data)
preds = [i > .50 for i in preds]


# In[ ]:


# Reading in test data target
test_target = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test_target.head()


# In[ ]:


# Results of predictions on test set
survived = test_target['Survived'].values
print(confusion_matrix(preds,survived))
print(classification_report(preds,survived))


# # Results
# - Not surprisingly, this simple neural network predicted on the test set with 85% overall accuracy. This is a simple example that speaks to the predictive power of deep learning.

# In[ ]:




