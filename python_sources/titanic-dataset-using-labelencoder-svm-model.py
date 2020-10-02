#!/usr/bin/env python
# coding: utf-8

# # Problem: To predict whether a passenger survived or not using Support Vector Machine (SVM)
# *And using uploaded titanic_dataset

# In[ ]:


# Import Libraries

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load Dataset From CSV 

titanic_data = pd.read_csv('../input/titanic-dataset/titanic_dataset.csv')
titanic_data.head()


# In[ ]:


# Get only the data (features) that you will use for analysis and prediction
# We are going to use 'Age', 'Sex','Pclass', 'Fare'

train_data = ['Age', 'Sex','Pclass', 'Fare', 'PassengerId' ]


# In[ ]:


# load your target data (y)

target_data = ['Survived']


# In[ ]:


# Put the columns together for analysis and prediction
X = titanic_data[train_data]
Y = titanic_data[target_data]


# In[ ]:


X.head()


# In[ ]:


# Identify data type

X['Sex'].dtype


# In[ ]:


X.Sex


# In[ ]:


Y.head()


# # Data Cleaning

# In[ ]:


#look for NaN Values

X['Pclass'].isnull().sum()


# In[ ]:


X['Fare'].isnull().sum()


# In[ ]:


X['Age'].isnull().sum()


# In[ ]:


X['Sex'].isnull().sum()


# In[ ]:


# Since 'Age' has significant number of null values but is an important feature, we are not able to drop this column, 
# But instead, fill Nan Values with 'median' 

X['Age'] = X['Age'].fillna(X['Age'].median())


# In[ ]:


X['Age'].isnull().sum()


# In[ ]:


X['Fare'] = X['Fare'].fillna(X['Fare'].median())


# In[ ]:


X['Fare'].isnull().sum()


# In[ ]:


# Convert 'Sex' from string into integer using LabelEncoder

le = LabelEncoder()


# In[ ]:


X['Sex'] = le.fit_transform(X['Sex'].astype(str))
X.head()


# # SPLIT TRAIN AND TEST DATA 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size= 0.80)


# In[ ]:


len(X_train) #how many data are for train


# In[ ]:


len(X_test) #how many data are for test


# # CREATE LINEAR SVM MODEL

# In[ ]:


from sklearn import svm
model_linearsvc = svm.LinearSVC()


# In[ ]:


#train the model
model_linearsvc.fit(X_train, y_train)


# In[ ]:


#check accuracy of model
model_linearsvc.score(X_test, y_test)


# # USING SVC MODEL

# In[ ]:


from sklearn.svm import SVC
model_svc = SVC()


# In[ ]:


model_svc.fit(X_train, y_train)


# In[ ]:


model_svc.score(X_test, y_test)


# # USING DECISION TREE

# In[ ]:


from sklearn import tree


# In[ ]:


model_tree = tree.DecisionTreeClassifier()


# In[ ]:


model_tree.fit(X_train, y_train)


# In[ ]:


model_tree.score(X_test, y_test)


# # USING REGRESSION MODEL

# In[ ]:


from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()


# In[ ]:


model_lr.fit(X_train, y_train)


# In[ ]:


model_lr.score(X_test, y_test)


# # USING RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()


# In[ ]:


model_rf.fit(X_train, y_train)


# In[ ]:


model_rf.score(X_test, y_test)


# # CONCLUSION
# * Since we see that Random Forest has the highest score for accuracy of prediction, then we use this model

# In[ ]:


# Output array([1]) means the person lived

model_rf.predict(X_test[0:1]) 


# In[ ]:


model_rf.predict(X_test[0:1]) 


# In[ ]:


# predict top 10 people of the test dataset

model_rf.predict(X_test[0:223]) 


# In[ ]:


y_test.head()


# In[ ]:


submission2 = pd.DataFrame({
        "PassengerId": X_test['PassengerId'],
        "Survived": model_rf.predict(X_test)
    })


# In[ ]:


submission2


# In[ ]:


submission2.to_csv('submission2.csv', index=False)

