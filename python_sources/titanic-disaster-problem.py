#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.info()


# In[ ]:


test['Sex'] = test['Sex'].replace ({'male':0, 'female':1})
test['Embarked'] = test['Embarked'].replace({'S':0, 'C':1, 'Q':2})


# In[ ]:


test.info()


# In[ ]:


test['Age'] = test['Age'].fillna(test['Age'].mean())


# In[ ]:


test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# In[ ]:


test.info()


# In[ ]:





# In[ ]:


sub.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.countplot(train['Sex'])


# In[ ]:


sns.countplot(train['Embarked'])


# In[ ]:


sns.countplot(train['Survived'])


# In[ ]:


train['Survived'].value_counts() # plotting survived Col


# In[ ]:


sns.countplot(train['Survived'], hue=train['Sex'])


# In[ ]:


sns.boxplot(train['Age'])


# In[ ]:


train[['Age']].describe()


# In[ ]:


train['Age']= train['Age'].fillna(train['Age'].mean())


# In[ ]:


train.info()


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
print(train['Embarked'].mode())


# In[ ]:


train.info()


# In[ ]:


X = train.drop(['Survived'], axis=1)
y = train[['Survived']]


# In[ ]:


X.head()


# In[ ]:


import seaborn as sns
sns.countplot(X['Embarked'])


# In[ ]:


y.head()


# In[ ]:


X= X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X.head()


# In[ ]:


X.info()


# In[ ]:


X['Sex']=X['Sex'].replace({'male':0, 'female':1})
X['Embarked'] = X['Embarked'].replace({'S':0, 'C':1, 'Q':2})


# In[ ]:





# In[ ]:


X.info()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print("X_train:"+str(X_train.shape))
print("X_test: "+str(X_test.shape))


# In[ ]:


y_pred_prob_train_lr = lr.predict_proba(X_train)
y_pred_prob_test_lr = lr.predict_proba(X_test)


# In[ ]:


lr= LogisticRegression()
lr.fit(X_train,y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(y_train, y_pred_train)
plot_confusion_matrix(y_test, y_pred_test)


# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_train, y_pred_train)
print(cr)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
n = np.arange(1,10)
sco_train =[]
sco_test=[]
for k in n:
    knn = KNeighborsClassifier(n_neighbors =k)
    knn.fit(X_train, y_train)
    sco_train.append(knn.score(X_train, y_train))
    sco_test.append(knn.score(X_test, y_test))


# In[ ]:


plt.plot(sco_train)
plt.plot(sco_test)
plt.legend(['Train', 'Test'])


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

plot_confusion_matrix(y_train, y_pred_train)
plot_confusion_matrix(y_test, y_pred_test)


# In[ ]:


from scikitplot.metrics import plot_roc
plot_roc( y_train, y_pred_prob_train)
plot_roc(y_test, y_pred_prob_test)


# In[ ]:


plot_roc(y_train, y_pred_prob_train_lr)


# In[ ]:


y_pred_prob_train = knn.predict_proba(X_train)
y_pred_prob_test=knn.predict_proba(X_test)


# In[ ]:


y_pred_prob_train


# In[ ]:





# In[ ]:


y_pred = lr.predict(test)


# In[ ]:


sub['pred'] = y_pred


# In[ ]:


sub.head()


# In[ ]:


sub.drop(['Survived'], axis=1, inplace=True)


# In[ ]:



sub.columns =['PassengerId','Survived']


# In[ ]:





# In[ ]:


sub


# In[ ]:


sub.to_csv("submission.csv", index = False)

