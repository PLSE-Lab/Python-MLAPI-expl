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


import sklearn # to apply logistic regression model
from sklearn.metrics import accuracy_score #to find the accuracy score of the model 


# In[ ]:


# Importing the dataset
dataset=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values


# In[ ]:


dataset


# In[ ]:


X


# In[ ]:


Y
#as you can see the species column is a categorical data so we are going to convert it into numerical form so that
#it can be fitted into model.For that we will use labelencoder from sklearn.preprocessing 


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
labelencoder_Y.fit_transform(Y)
Y=labelencoder_Y.fit_transform(Y)
Y


# In[ ]:


#now we are going to split the dataset into train set and test set and use train set pair to make model learn
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[ ]:


#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
    


# In[ ]:


#predicting the test result
y_pred=classifier.predict(X_test)


# In[ ]:


#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
cm


# In[ ]:


#as you can see in the above confusion matrix there are 13+15+9 true positive predictions 
#now we will calculate accuracy of the model

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(Y_test,y_pred))
# I have used KNN model on it too but it gave me same accuracy rate on my machine.

