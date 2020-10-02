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


# In[ ]:


titanic_data=pd.read_csv("../input/train.csv")

titanic_data.head(10)


# In[ ]:


titanic_data.columns


# In[ ]:


titanic_data.isnull().sum()


# In[ ]:


# removing null values of age and embarked

titanic_data['Embarked']=titanic_data['Embarked'].replace(np.nan,'S')
median=titanic_data['Age'].median()
median
titanic_data['Age']=titanic_data['Age'].replace(np.nan,median)


# In[ ]:


titanic_data.isnull().sum()


# In[ ]:


# dealing with categorial values -----converting into numerical values by using labelencoder
from sklearn import preprocessing
le_sex=preprocessing.LabelEncoder()
le_sex.fit(titanic_data.Sex.unique())
titanic_data.Sex=le_sex.transform(titanic_data.Sex)

from sklearn import preprocessing
le_Embarked=preprocessing.LabelEncoder()
le_Embarked.fit(titanic_data.Embarked.unique())
titanic_data.Embarked=le_Embarked.transform(titanic_data.Embarked)

from sklearn import preprocessing
le_Pclass=preprocessing.LabelEncoder()
le_Pclass.fit(titanic_data.Pclass.unique())
titanic_data.Pclass=le_Pclass.transform(titanic_data.Pclass)


# 
# 
# 

# In[ ]:


titanic_data.head()


# 

# In[ ]:


#dropping ["PassengerId","Name","Survived","Ticket","Cabin"] columns as they dont effect surviving people

X=titanic_data.drop(["PassengerId","Name","Survived","Ticket","Cabin"],axis=1)
X.isnull().sum()
X.head()


# In[ ]:


Y=titanic_data.iloc[:,1]

Y.isnull().sum()
Y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=20)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train, Y_train)

predictions=logmodel.predict(X_test)


# In[ ]:




from sklearn.metrics import classification_report

classification_report(Y_test, predictions)



# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,predictions)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(Y_test,predictions)

