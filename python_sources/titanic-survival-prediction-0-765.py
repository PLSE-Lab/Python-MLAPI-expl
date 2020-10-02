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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.columns.values


# In[ ]:


#removing the unnecessary columns
train_df = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis = 1)
test_df = test.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis = 1)


# In[ ]:


train_df.info()


# In[ ]:


#we have null values in "Age", so we fill those by Imputer

from sklearn.impute import SimpleImputer
num = train_df.select_dtypes(exclude=['object'])
my_imputer = SimpleImputer()
imputed_train = pd.DataFrame(my_imputer.fit_transform(num))
imputed_train.columns = num.columns
imputed_train


# In[ ]:


imputed_train.info()
#Alright, no null values now


# In[ ]:


#Checking any null categorical Values
obj = train_df.select_dtypes(exclude = ['int64','float'])
obj.info()


# In[ ]:


#fill it by max used value 
freq = obj.Embarked.dropna().mode()[0]
freq


# In[ ]:


obj['Embarked'] = obj['Embarked'].fillna(freq)
obj.info()
#No Null Values 


# In[ ]:


#Converting Categorical values to Numerical
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_train = pd.DataFrame(OH_encoder.fit_transform(obj))
OH_train.index = obj.index
OH_train


# In[ ]:


train_fill = pd.concat([imputed_train, OH_train], axis = 1)
train_fill


# In[ ]:


train_fill.info()


# In[ ]:


#Applying all the above methods for test file
train_df.info()


# In[ ]:


#we have null values only in "Age" column
num_test = test_df.select_dtypes(exclude=['object'])
my_imputer = SimpleImputer()
imputed_test = pd.DataFrame(my_imputer.fit_transform(num_test))
imputed_test.columns = num_test.columns
imputed_test


# In[ ]:


obj_test = test_df.select_dtypes(exclude = ['int64','float'])
obj_test.info()
#No Null values = All Happies


# In[ ]:


#Converting Categorical values to Numerical
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_test = pd.DataFrame(OH_encoder.fit_transform(obj_test))
OH_test.index = obj_test.index
OH_test


# In[ ]:


test_fill = pd.concat([imputed_test, OH_test], axis = 1)
test_fill


# In[ ]:


Y_train = train_fill['Survived']
Y_train


# In[ ]:


X_train = train_fill.drop(['Survived'], axis = 1)
X_train


# In[ ]:


#Let us find the scores from different models and submit the best one
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(test_fill).astype(int)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(test_fill)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(test_fill)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_final = logreg.predict(test_fill).astype(int)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(test_fill)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(test_fill)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(test_fill)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(test_fill)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test_fill).astype(int)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


from xgboost import XGBClassifier 
my_model = XGBClassifier()
my_model.fit(X_train, Y_train)
predictions = my_model.predict(test_fill).astype(int)
acc_xgb = round(my_model.score(X_train, Y_train) * 100, 2)
acc_xgb


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_final
    })


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




