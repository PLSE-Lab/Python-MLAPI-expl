# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
print ('hello')
data=pd.read_csv("../input/Employeechurn.csv")
data.head()
data.salary=data.salary.astype('category')
data.salary=data.salary.cat.codes
departments=pd.get_dummies(data.department)
n_employees=len(data)
from sklearn.model_selection import train_test_split
target=data.churn
features=data
features.pop('churn')
features.pop('department')
target_train, target_test,features_train,feature_test=train_test_split(target,features,test_size=0.25)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
features_train
model.fit(features_train, target_train)
model.score(features_train,target_train)*100