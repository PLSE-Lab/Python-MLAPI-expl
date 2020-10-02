#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > ## Loading Data in python

# In[ ]:


df = pd.read_csv("../input/train.csv")
print(df.head())


# > ## Dropping unecessary columns

# In[ ]:


y = df['Survived']
X=df.drop(['Name','Ticket','Survived','PassengerId','Cabin'],axis=1)


# > ## Getting dummy columns

# In[ ]:


X=pd.get_dummies(X,columns=['Sex','Embarked'])


# > ## Creating age categories

# In[ ]:


def age_div(x):
    if x>=0 and x<=8:
        return 'Infant'
    elif x>8 and x<=18:
        return 'Children'
    elif x>18 and x<=50:
        return 'Adult'
    elif x>50:
        return 'Old'
    else:
        return 'NoEntry'
X['Age_new']=X['Age'].apply(age_div)


# In[ ]:


X=pd.get_dummies(X,columns=['Age_new'])


# In[ ]:


X.drop(['Age'],axis=1,inplace=True)


# In[ ]:


X.info()


# >> ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# > ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


clf = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=250)


# In[ ]:


clf.fit(X_train,y_train)


# >> ## Predicting Test Set

# In[ ]:


y_predict = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


acc = accuracy_score(y_test, y_predict)


# In[ ]:


print("The accuracy is ",acc*100, "%.")


# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


X=df_test.drop(['Name','Ticket','PassengerId','Cabin'],axis=1)


# In[ ]:


def age_div(x):
    if x>=0 and x<=8:
        return 'Infant'
    elif x>8 and x<=18:
        return 'Children'
    elif x>18 and x<=50:
        return 'Adult'
    elif x>50:
        return 'Old'
    else:
        return 'NoEntry'
X['Age_new']=X['Age'].apply(age_div)


# In[ ]:


X=pd.get_dummies(X,columns=['Sex','Embarked','Age_new'])


# In[ ]:


X.drop(['Age'],axis=1,inplace=True)


# In[ ]:


X.info()


# In[ ]:


X=X.fillna(method='ffill')


# In[ ]:


y_pred=clf.predict(X)


# In[ ]:


serial = df_test['PassengerId']
data = {'PassengerId': df_test['PassengerId'], 'Survived': y_pred}
submission = pd.DataFrame(data)
submission.to_csv('Submission.csv', index=False)


# In[ ]:




