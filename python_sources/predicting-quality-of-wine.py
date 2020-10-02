#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you writee to the current directory are saved as output.


# In[ ]:


#loading data in to csv
data = pd.read_csv("../input/winequality-red.csv")
# viewing top 10 rows
data.head(10)


# In[ ]:


#viewing information of data features
data.info()


# In[ ]:


#describe data
data.describe()


# In[ ]:


# couting number of types of quality
data['quality'].value_counts()


# In[ ]:


data['quality'].value_counts().plot.bar()


# In[ ]:


data.corr()


# In[ ]:


data.head(10)


# In[ ]:


data['quality'] = data['quality'].astype(int)


# In[ ]:


bins = (2, 6, 8)
group_names = ['bad','good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names, include_lowest = False)


# In[ ]:


data.head(10)
#quality is divided in binary classification from multilabel classification


# In[ ]:


#couting the number of bad and good value
data['quality'].value_counts()


# In[ ]:


#assign label to quality
from sklearn.preprocessing import LabelEncoder
qual = LabelEncoder()
data['quality']=qual.fit_transform(data['quality'])


# In[ ]:


#after fitting encoding bad becomes 0 and good becomes 1
data['quality'].value_counts()


# In[ ]:


#separate dataset as features and target
y = data['quality']
X = data.drop('quality',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


# In[ ]:


#splitting data set into  train and test
train_x,test_x,train_y,test_y = train_test_split(X,y ,test_size =0.2,random_state= 42)


# In[ ]:


#performing standard scaling
scal = StandardScaler()
train_x = scal.fit_transform(train_x)
test_x = scal.fit_transform(test_x)


# In[ ]:


#fitting the model and predicting result
forest = RandomForestClassifier(n_estimators=400,random_state = 42)
forest.fit(train_x,train_y)
predicts = forest.predict(test_x)


# In[ ]:


#confusion matrix 
confusionMatrix = confusion_matrix(test_y,predicts)
confusionMatrix


# In[ ]:


print(classification_report(test_y,predicts))


# In[ ]:


svc1 = SVC()
svc1.fit(train_x,train_y)
pred_svc = svc1.predict(test_x)


# In[ ]:


#Finding best parameters for svc
parameters = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid = GridSearchCV(svc1, param_grid=parameters, scoring='accuracy', cv=10)
grid.fit(train_x,train_y)
print(grid.best_params_)


# In[ ]:


#try SVC for best parameters
svc = SVC(kernel = 'rbf',random_state = 42,gamma= 0.9,C=1.2)
svc.fit(train_x,train_y)
pred = svc.predict(test_x)


# In[ ]:


print(classification_report(test_y,pred))
print(confusion_matrix(test_y,pred))


# In[ ]:


#we get 90% using best parameters for SVC.

