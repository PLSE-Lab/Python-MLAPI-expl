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





# In[ ]:


patient= pd.read_csv("../input/indian_liver_patient.csv")
patient.head()


# In[ ]:





# In[ ]:


pd.get_dummies(patient['Gender'], prefix = 'Gender').head()


# In[ ]:





# In[ ]:


patient = pd.concat([patient,pd.get_dummies(patient['Gender'], prefix = 'Gender')], axis=1)


# In[ ]:





# In[ ]:


patient["Albumin_and_Globulin_Ratio"] = patient.Albumin_and_Globulin_Ratio.fillna(patient['Albumin_and_Globulin_Ratio'].mean())


# In[ ]:





# In[ ]:


X = patient.drop(['Gender','Dataset'], axis=1)


# In[ ]:





# In[ ]:


y = patient['Dataset']


# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[ ]:





# In[ ]:



random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
print(confusion_matrix(y_test,rf_predicted))
print("Classification Report:\n{}".format(classification_report(y_test, rf_predicted)))


# In[ ]:


X.head()


# In[ ]:


exdata=[[65,2,1.1,300,25,22,5.5,3.3,0.8,0,1],[65,2,1.1,300,25,22,5.5,3.3,0.8,1,0],[17,0.9,0.3,202,22,19,7.4,4.1,1.2,0,1]]


# In[ ]:


ydata=random_forest.predict(exdata)


# In[ ]:


print(ydata)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print(ydata)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




