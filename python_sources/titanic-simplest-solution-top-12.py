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


data_set = pd.read_csv("/kaggle/input/titanic/train.csv")
data_set_test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


data_set['Embarked'] = data_set['Embarked'].fillna("S")
data_set_test['Embarked'] = data_set_test['Embarked'].fillna("S")


# In[ ]:


X_train = data_set.iloc[:,[2, 4, 5, 6, 7, 9, 11]].values
Y_train = data_set.iloc[:,1].values
X_test = data_set_test.iloc[:,[1, 3, 4, 5, 6, 8, 10]].values
pid_col = data_set_test.iloc[:, 0].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_train[:, 6] = labelencoder_X.fit_transform(X_train[:, 6])

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6])


# In[ ]:


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train[:,:] = imp.fit_transform(X_train[:,:])
X_test[:,:] = imp.fit_transform(X_test[:,:])


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder2 = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder2.fit_transform(X_test).toarray()


# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate=0.001,n_estimators=1000)
classifier.fit(X_train, Y_train)


# In[ ]:


Y_pred = classifier.predict(X_test)


# In[ ]:


ans =pd.DataFrame({'PassengerId':pid_col, 'Survived':Y_pred})
ans.to_csv("titanicPred.csv", index=False)


# In[ ]:


ans


# In[ ]:




