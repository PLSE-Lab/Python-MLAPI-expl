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


import pandas as pd
titanic_train=pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


titanic_train.head()


# In[ ]:


#check null values
titanic_train.isnull().sum()


# We can see that Age have 177 values are null,cabin have large values are null

# In[ ]:


#Lets remove Cabin and Embarked 
titanic_train.drop(['Cabin','Embarked'],axis=1,inplace=True)


# In[ ]:


# Lets impute the null values with mean or median
titanic_train.describe()


# In[ ]:


titanic_train.Age.value_counts()


# We see more number of occurances of age is between 24 to 30 so lets impute randomly from 24 to 30

# In[ ]:


import random
titanic_train['Age']=titanic_train['Age'].fillna(random.randint(24,31))


# In[ ]:


#again check for null values
titanic_train.isnull().sum()


# In[ ]:


#Let's convert the categorical value of Sex to binary[0,1]
titanic_train.Sex.replace(['male','female'],[0,1],inplace=True)


# In[ ]:


titanic_train.head()


# In[ ]:


#let's remove name and ticket where these variables don't significantly explain the data
titanic_train.drop(['Ticket','Name'],axis=1,inplace=True)


# In[ ]:


titanic_train.head()


# In[ ]:


X_train=titanic_train.drop('Survived',axis=1)
y_train=titanic_train.Survived


# In[ ]:


#lets scale the variables of titanic_train
from sklearn.preprocessing import StandardScaler


# In[ ]:


scale=StandardScaler()
X_train[X_train.columns]=scale.fit_transform(X_train)


# In[ ]:


X_train.corr()


# In[ ]:


print(titanic_train.head())
print(X_train.head())


# In[ ]:


#Lets apply RFE(recursive feature elimination)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()


# In[ ]:


rfe=RFE(model,6)
X_train_rfe=rfe.fit(X_train,y_train)


# In[ ]:


list(zip(X_train.columns, X_train_rfe.support_, X_train_rfe.ranking_))


# In[ ]:


X_rfe=X_train.columns[X_train_rfe.support_]
X_rfe_train=X_train[X_rfe]


# In[ ]:


X_rfe_train.head()


# In[ ]:


model.fit(X_rfe_train,y_train)


# In[ ]:


y_pred=model.predict(X_rfe_train)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.accuracy_score(y_train,y_pred)


# In[ ]:


confusion=metrics.confusion_matrix(y_train,y_pred)
print(confusion)


# In[ ]:


metrics.mean_squared_error(y_train,y_pred)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train, y_pred, drop_intermediate = False )
print(fpr , tpr , thresholds)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


def draw_roc( actual, pred ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, pred,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, pred )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


draw_roc(y_train,y_pred)


# #lets predict on test data

# In[ ]:


titanic_test['Age']=titanic_test['Age'].fillna(random.randint(24,30))
titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test.Fare.mean())


# In[ ]:


x_test=titanic_test[X_train.columns]


# In[ ]:


x_test.Sex.replace(['male','female'],[0,1],inplace=True)


# In[ ]:


x_test.head()


# In[ ]:


x_test[x_test.columns]=scale.transform(x_test)


# In[ ]:


x_test.head()


# In[ ]:


x_test.isnull().sum()


# In[ ]:


x_test.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


y_test=model.predict(x_test)


# In[ ]:


y_test


# In[ ]:


len(y_test)


# In[ ]:


submission=pd.DataFrame({'PassengerId': titanic_test.PassengerId,'Survived':y_test})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('survival predictions',index=False)

