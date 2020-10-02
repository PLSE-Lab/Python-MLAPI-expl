#!/usr/bin/env python
# coding: utf-8

# Hello kagglers,In this Kernel, I have used XGBClassifier is trained on complete training data to learn more about data and this always increases the accuracy by a large margin. The classifer's parameter is obtained using GridSearchCV.
# Your feedback is important. :)
# If you like the work Please **UpVote** :)

# In[ ]:


# Importing modules
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


# Loading Datasets
data_train  = pd.read_csv('../input/learn-together/train.csv')
data_test = pd.read_csv('../input/learn-together/test.csv')


# In[ ]:


# Droping 'Id' and 'Cover_Type' columns
data_train=data_train.drop(['Id'],axis=1)
data_id = data_test['Id']
data_test = data_test.drop(['Id'],axis=1)

x = data_train.drop(['Cover_Type'],axis=1)
y = data_train['Cover_Type']


# In[ ]:


# The model parameters are obtained using GridSearchCV and insted spliting the dataset into train and 
# valid sets I am using the complete training set to Train.
clf = XGBClassifier(n_estimators=500,colsample_bytree=0.9,max_depth=9,random_state=1,eta=0.2)
clf.fit(x,y)


# In[ ]:


test_pred = clf.predict(data_test)
output = pd.DataFrame({'Id': data_id,
                       'Cover_Type': test_pred})
output.to_csv('submission.csv', index=False)

