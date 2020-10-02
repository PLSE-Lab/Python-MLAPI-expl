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


#Import the accuracy score function
from sklearn.metrics import accuracy_score


# In[ ]:


data = pd.read_csv("../input/train.csv", low_memory = False)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, oob_score=True)


# In[ ]:


X = pd.get_dummies(data[['ZIP','rent', 'education', 'income','loan_size', 'payment_timing', 'job_stability', 'occupation']])


# In[ ]:


y = data.default


# In[ ]:


clf.fit(X, y)


# In[ ]:


y_train = clf.predict(X)


# In[ ]:


test_data = pd.read_csv("../input/test.csv", low_memory = False)


# In[ ]:


#create dummies for all features in test data and call X_test
X_test = pd.get_dummies(test_data[['ZIP','rent', 'education', 'income','loan_size', 'payment_timing', 'job_stability', 'occupation']])


# In[ ]:


#Predict y_pred using the classifier trained on the training set

y_pred = clf.predict (X_test)


# In[ ]:


#create a y for the test set
y_test = test_data.default


# In[ ]:


accuracy_score (y_test,y_pred)


# In[ ]:


# Creates new column that shows if the model predicts default or not, title "y_pred"
test_data['default_pred'] = (y_pred)


# In[ ]:


accuracy_score (y_test,y_pred)


# In[ ]:


minority_groups_test = test_data.groupby("minority")
min_cant_pay = minority_groups_test["default"].sum()
print(min_cant_pay)


# In[ ]:


min_cant_pay_pred = minority_groups_test["default_pred"].sum()
print(min_cant_pay_pred)


# In[ ]:


true_positive_minority = 100*(1 - (min_cant_pay_pred/min_cant_pay))
print(true_positive_minority)


# In[ ]:


sex_groups_test = test_data.groupby("sex")
sex_cant_pay = sex_groups_test["default"].sum()
print(sex_cant_pay)


# In[ ]:


sex_cant_pay_pred = sex_groups_test["default_pred"].sum()
sex_groups_test = test_data.groupby("sex").count()
pred_non_default = sex_groups_test - sex_cant_pay
print(pred_non_default)
print(sex_cant_pay_pred)


# In[ ]:


true_positive_sex = 100*(1 - (sex_cant_pay_pred/sex_cant_pay))
print(true_positive_sex)


# In[ ]:





# In[ ]:


#Question 12 - Is the loan granting scheme equal opportunity? 
#Compare the share of successful non-minority applicants that defaulted to the share of successful minority applicants that defaulted. 
minority_groups = data.groupby("minority")
print("Original Data Default grouped by minority:")
print(minority_groups["default"].count())

minority_groups_test = test_data.groupby("minority")
print("Test Data Default grouped by minority:")
print(minority_groups_test["default"].count())

#Do the same comparison of the share of successful female applicants that default versus successful male applicants that default
sex_groups = data.groupby("sex")
print("Original Data Default grouped by sex:")
print(sex_groups["default"].count())

sex_groups_test = test_data.groupby("sex")
print("Test Data Default grouped by sex:")
print(sex_groups_test["default"].count())

#The result of the comparisons show that there is no relation between either sex or minority status 
#on default rates if anything they are equivilant

