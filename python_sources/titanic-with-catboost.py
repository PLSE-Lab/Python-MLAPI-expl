#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Load Train and test data
X_train = pd.read_csv('/kaggle/input/titanic/train.csv')
X_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


#Get a brief about the data types
X_train.info()


# In[ ]:


X_train.head()


# In[ ]:


#The percentage of null values in the dataset
print(X_train.isnull().sum()/X_train.shape[0])
print('*'*25)
print(X_test.isnull().sum()/X_test.shape[0])


# In[ ]:


#More than 70 % of data is missing in cabin column . We can drop it.
X_train.drop(['Cabin'],axis=1,inplace=True)
X_test.drop(['Cabin'],axis=1,inplace=True)

#Impute the mising values in Age and Fare
X_train['Age'].fillna(X_train['Age'].median(),inplace=True)
X_test['Age'].fillna(X_test['Age'].median(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)
#There are only three missing values in Embarked column, so we can drop it
X_train = X_train.dropna(axis=0, subset=['Embarked'])


# In[ ]:


#Drop the passengerid as it is just a unique identifier and also drop dependant variable
X = X_train.drop(['Survived','PassengerId'],axis=1)
y= X_train['Survived']

#Split the into train and validation and use test data for predictions
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=10)


# In[ ]:


#Find the indices for categorical columns
categorical_features_indices = np.where(X_train.dtypes == np.object)[0]


# In[ ]:


#importing library and building model. 
#Here we are using CatBoostRegressor as our model. CatBoostRegressor will handle categorical data
#I have fine tuned the hyperparameters -depth,iterations,learning_rate for this dataset
from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=34, depth=15, learning_rate=0.2, loss_function='Logloss')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_val, y_val),plot=True)


# In[ ]:


#Catboosregressor gives the probabiites rather than the actual value.
y_pred_train = pd.Series(model.predict(X)) 
X_test_1 = X_test.drop('PassengerId',axis=1) 
y_pred = pd.Series(model.predict(X_test_1)) 
y_pred_val = pd.Series(model.predict(X_val))
def step(x): #Step function
    if x> 0.5:
        return 1
    else:
        return 0

#Apply step function to convert the predictions to 0 or 1
y_pred_train = y_pred_train.apply(step)
y_pred = y_pred.apply(step)
y_pred_val = y_pred_val.apply(step)


# In[ ]:


#Create confusion matrix and compute the accuracy score of predictions
from sklearn.metrics import confusion_matrix,accuracy_score
cm_train = confusion_matrix(y,y_pred_train)
ac = accuracy_score(y,y_pred_train)
print("Confucion matirx on train data\n",cm_train)
print("Accuracy score on train data: ",ac)


# In[ ]:


print("Accuracy score on validation data: ",accuracy_score(y_val,y_pred_val))


# In[ ]:


#Prepare for submission
X_test_predict = X_test.drop('PassengerId',axis=1) #drop the passenger id as we did't fit the model with passengerid column
predict_to_submit = pd.Series(model.predict(X_test_predict)) #make predcitions and convert them into padnaseries so that we can apply step function on the pandas series data
predict_to_submit = predict_to_submit.apply(step) #Apply step function to convert the predictions to 0 or 1
my_submission = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': predict_to_submit}) #Make the DF for submision data
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False) #Save the submission file.


# In[ ]:




