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


# # Importing Data

# In[ ]:


import pandas as pd

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


titanic_train.head()


# In[ ]:


titanic_train.describe()


# In[ ]:


titanic_train.isnull().sum()


# In[ ]:


titanic_train.Name.value_counts()


# In[ ]:


titanic_train.Ticket.value_counts()


# In[ ]:


titanic_train.Cabin.value_counts()


# In[ ]:


new_train = titanic_train.drop(['Name','Ticket','Cabin'],axis=1)
new_train


# In[ ]:


new_train['Age']=new_train['Age'].fillna(new_train['Age'].mean())
new_train.isnull().sum()


# In[ ]:


new_train['family_size'] = new_train['SibSp']+new_train['Parch']
new_train.family_size.value_counts()


# In[ ]:


new_train['FareBin'] = pd.qcut(new_train['Fare'], 4)
new_train['AgeBin'] = pd.qcut(new_train['Age'], 4)


# In[ ]:


new_train = new_train.drop(['Fare','Age'],axis=1)
new_train = new_train.dropna(subset=['Embarked'])
new_train


# In[ ]:


categorical_features = new_train.drop(['PassengerId','Survived','SibSp','Parch','family_size'],axis=1)


# In[ ]:


categorical_features


# In[ ]:


a = categorical_features.columns.tolist()
encoded_features = pd.get_dummies(categorical_features, columns=a)
encoded_features


# In[ ]:


esc_dummy_trap = encoded_features.drop(['Pclass_3','Sex_male','Embarked_S','FareBin_(31.0, 512.329]','AgeBin_(35.0, 80.0]'],axis=1)
esc_dummy_trap


# In[ ]:


X = new_train.drop(['Pclass','Sex','Embarked','FareBin','AgeBin'],axis=1)
X= pd.concat([X,esc_dummy_trap],axis=1)
X


# In[ ]:


X_train = X.drop(['Survived','PassengerId'],axis=1)
X_train


# In[ ]:


Y_train = X['Survived']
Y_train.dtype


# In[ ]:


X_train.isnull().sum()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# In[ ]:


Y_train_pred = classifier.predict(X_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(classifier, X_train, Y_train, cv=3, scoring="accuracy")


# In[ ]:


corr_matrix = X.corr()
corr_matrix['Survived'].sort_values(ascending=False)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(Y_train,Y_train_pred)
cm


# In[ ]:


acc_scr = accuracy_score(Y_train,Y_train_pred)
acc_scr


# In[ ]:


report = classification_report(Y_train, Y_train_pred)
print(report)


# # **test set**

# In[ ]:


titanic_test


# In[ ]:


for i in titanic_test['Parch']:
    titanic_test['Parch'] = titanic_test['Parch'].replace(9, 6)


# In[ ]:


new_test = titanic_test.drop(['Name','Ticket','Cabin'],axis=1)


# In[ ]:


new_test['Age']=new_test['Age'].fillna(new_test['Age'].mean())
new_test.isnull().sum()


# In[ ]:


new_test['Fare']=new_test['Fare'].fillna(new_test['Fare'].mean())
new_test.isnull().sum()


# In[ ]:


new_test['family_size'] = new_test['SibSp']+new_test['Parch']
new_test.family_size.value_counts()


# In[ ]:


new_test['FareBin'] = pd.qcut(new_test['Fare'], 4)
new_test['AgeBin'] = pd.qcut(new_test['Age'], 4)
new_test = new_test.drop(['Age','Fare'],axis=1)


# In[ ]:


titanic_test.Parch.value_counts()


# In[ ]:


categorical_features_test = new_test.drop(['PassengerId','SibSp','Parch','family_size'],axis=1)
categorical_features_test


# In[ ]:


a_test = categorical_features_test.columns.tolist()
encoded_features_test = pd.get_dummies(categorical_features_test, columns=a_test)
encoded_features_test


# In[ ]:


esc_dummy_trap_test = encoded_features_test.drop(['Pclass_3','Sex_male','Embarked_S','FareBin_(31.5, 512.329]','AgeBin_(35.75, 76.0]'],axis=1)


# In[ ]:


new_test


# In[ ]:


X_test = new_test.drop(['Pclass','Sex','Embarked','FareBin','AgeBin'],axis=1)
X_test = pd.concat([X_test,esc_dummy_trap_test],axis=1)


# In[ ]:


X_test = X_test.drop(['PassengerId'],axis=1)


# In[ ]:


X_test


# In[ ]:


X_train


# In[ ]:


titanic_test.Parch.value_counts()


# In[ ]:


test_pred = classifier.predict(X_test)


# **Submitting Output**

# In[ ]:


output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': test_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

