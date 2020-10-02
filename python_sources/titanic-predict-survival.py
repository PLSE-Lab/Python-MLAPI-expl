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


dataset = pd.read_csv("../input/train.csv")
dataset.isnull().any()


# In[ ]:


y = dataset.iloc[:, 1].values
dataset.drop(["Survived"], axis = 1, inplace = True)
#dataset.describe()
dataset.drop(["PassengerId","Name","Ticket","Cabin","Embarked"], axis = 1, inplace = True)


# In[ ]:


dataset.head()


# In[ ]:


X = dataset.iloc[:,:].values


# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])


# In[ ]:


X


# In[ ]:


X_dataframe = pd.DataFrame(X)


# In[ ]:


X_dataframe


# In[ ]:


dataset


# In[ ]:


X_dataframe = pd.DataFrame(X,columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])


# In[ ]:


X_dataframe


# In[ ]:


X_dataframe.isnull().any()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])


# In[ ]:


X


# In[ ]:


X_dataframe = pd.DataFrame(X,columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])


# In[ ]:



X_dataframe


# In[ ]:


X_train = X
y_train = y


# In[ ]:


dataset = pd.read_csv("../input/test.csv")
dataset.isnull().any()
dataset.drop(["PassengerId","Name","Ticket","Cabin","Embarked"], axis = 1, inplace = True)
dataset.isnull().any()


# In[ ]:


dataset


# In[ ]:


X = dataset.iloc[:,:].values


# In[ ]:


X


# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
imputer2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer2.fit(X[:, 5:])
X[:, 5:] = imputer2.transform(X[:, 5:])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

X_dataframe = pd.DataFrame(X,columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])
X_dataframe.isnull().any()

X_dataframe.head()


# In[ ]:


X_test = X


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


Y_pred = np.around(y_pred)
Y_pred = Y_pred.astype(int)


# In[ ]:


result = pd.DataFrame(Y_pred)


# In[ ]:


result.astype(int)


# In[ ]:


submit_example = pd.read_csv("../input/gender_submission.csv")


# In[ ]:


submit_example


# In[ ]:


submit = submit_example.iloc[:,:].values
submit[:,1] = Y_pred
submission = pd.DataFrame(submit, columns=["PassengerId","Survived"])


# In[ ]:


submission


# In[ ]:


submission.to_csv('sub.csv', index=False)


# In[ ]:




