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


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


import os
os.chdir(r'/kaggle/working')


# # **EDA :**

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


survived = train[train["Survived"]==1]["Sex"].value_counts()
survived
dead = train[train["Survived"]==0]['Sex'].value_counts()
dead
df = pd.DataFrame([survived,dead])
df


# In[ ]:


def bar_chart(param):
    survived = train[train["Survived"]==1][param].value_counts()
    dead = train[train["Survived"]==0][param].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ["Survived","Dead"]
    df.plot(kind="bar", figsize = (10,5))


# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('Embarked')


# In[ ]:


bar_chart('Pclass')


# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# In[ ]:


test['Survived']=""
test.head()


# In[ ]:


train.info()


# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace = True)

    #complete embarked with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
train.info()


# In[ ]:


test.info()


# In[ ]:


test['Age'].fillna(test['Age'].median(), inplace = True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)
test.info()


# In[ ]:


train = train.drop(['Cabin','Name','Ticket'],axis=1)
test = test.drop(['Cabin','Name','Ticket'],axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train['Sex'] = lb_make.fit_transform(train['Sex'])
train['Embarked'] = lb_make.fit_transform(train['Embarked'])

test['Sex'] = lb_make.fit_transform(test['Sex'])
test['Embarked'] = lb_make.fit_transform(test['Embarked'])
train.head()


# In[ ]:


plt.figure(figsize=(20,10))
c= train.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In[ ]:


X_train = train[['Pclass','SibSp','Sex','Age','Embarked']]
y_train = train['Survived']
X_test = test[['Pclass','SibSp','Sex','Age','Embarked']]
y_test = test['Survived']


# # 1- K-neighbors :

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
"""print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))"""


# In[ ]:


test["Survived"] = knn.predict(X_test)
test


# In[ ]:


df = test[["PassengerId","Survived"]]
df


# In[ ]:


df.to_csv('submission.csv')


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')


# # 2- Logistic Regression :

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
test["Survived"] = logreg.predict(X_test)
df = test[["PassengerId","Survived"]]
df.to_csv('submission.csv')
from IPython.display import FileLink
FileLink(r'submission.csv')


# # 3- Decision Tree :

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
test["Survived"] = clf.predict(X_test)
df = test[["PassengerId","Survived"]]
df.to_csv('submission.csv')
from IPython.display import FileLink
FileLink(r'submission.csv')


# # 4- Linear Discriminant Analysis :

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
test["Survived"] = lda.predict(X_test)
df = test[["PassengerId","Survived"]]
df.to_csv('submission.csv')
from IPython.display import FileLink
FileLink(r'submission.csv')


# # 5- Gaussian Naive Bayes :
# 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
test["Survived"] = gnb.predict(X_test)
df = test[["PassengerId","Survived"]]
df.to_csv('submission.csv')
from IPython.display import FileLink
FileLink(r'submission.csv')


# # 6- Support Vector Machine SVM :

# In[ ]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
test["Survived"] = svm.predict(X_test)
df = test[["PassengerId","Survived"]]
df.to_csv('submission.csv')
from IPython.display import FileLink
FileLink(r'submission.csv')


# # 7- Random Forest :

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
test["Survived"] = rf.predict(X_test)
df = test[["PassengerId","Survived"]]
df.to_csv('submission.csv')

