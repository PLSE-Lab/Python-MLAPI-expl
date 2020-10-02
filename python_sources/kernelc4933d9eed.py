#!/usr/bin/env python
# coding: utf-8

# In[8]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
X_test = pd.read_csv("../input/test.csv", index_col="PassengerId")
data = pd.read_csv("../input/train.csv", index_col="PassengerId")



# In[9]:


import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

data.dropna(axis= 0, subset=["Survived"], inplace=True)

# get the target
y = data.Survived
data.drop(axis=1,columns="Survived", inplace=True)

#get important features
important_feature = ["Pclass", "Sex", "Age", "Fare"]
X = data[important_feature].copy()

train_X_non_final, valid_X_non_final, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


numerical_col =[col for col in X.columns if X[col].dtype != "object"]
categorical_col =[col for col in X.columns if X[col].dtype == "object"]

#deal with missing values
from sklearn.impute import  SimpleImputer
si = SimpleImputer(strategy="median")

X_train = pd.DataFrame(si.fit_transform(train_X_non_final[numerical_col]))
X_train.index = train_X_non_final.index
X_train.insert(1, "Sex", train_X_non_final.Sex)
X_train.columns = train_X_non_final.columns

X_valid = pd.DataFrame(si.transform(valid_X_non_final[numerical_col]))
X_valid.index = valid_X_non_final.index

X_valid.insert(1, "Sex", valid_X_non_final.Sex)
X_valid.columns = valid_X_non_final.columns

#deal with Categorical Variables
oh_train = pd.get_dummies(X_train[categorical_col])
X_train.drop(axis=1, columns="Sex",inplace=True)
X_train = pd.concat([X_train, oh_train], axis=1)

oh_valid = pd.get_dummies(X_valid[categorical_col])
X_valid.drop(axis=1, columns="Sex", inplace=True)
X_valid = pd.concat([X_valid, oh_valid], axis=1)


#build model and train
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

#predict using Evaluation set
y_pred = model.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred=y_pred, y_true=y_valid))

#preidct using Test set
X_test = X_test[important_feature].copy()
imputed_test_x = pd.DataFrame(si.transform(X_test[numerical_col]))
imputed_test_x.index = X_test.index

imputed_test_x.insert(1, "Sex", X_test.Sex)
imputed_test_x.columns = X_test.columns
X_test = imputed_test_x

oh_test = pd.get_dummies(X_test[categorical_col])
X_test.drop(axis=1, columns = "Sex", inplace=True)
X_test = pd.concat([X_test, oh_test], axis=1)
y_pred = model.predict(X_test)

output = pd.DataFrame({"PassengerId": X_test.index, "Survived":y_pred})
output.to_csv("submission.csv", index = False)


# In[ ]:




