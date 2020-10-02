#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/busara.csv')


# In[ ]:


data.head()


# In[ ]:


data['survey_date'] = pd.to_datetime(data['survey_date'])
data['month'] = data['survey_date'].dt.strftime('%m')
data['year'] = data['survey_date'].dt.strftime('%Y')
data['day'] = data['survey_date'].dt.strftime('%w')


# In[ ]:


data.head()


# In[ ]:


data['year'] = pd.get_dummies(data['year'])


# In[ ]:


data.head()


# In[ ]:


s  = ['survey_date']
data = data.drop(s,axis=True)


# In[ ]:


data = data.dropna(axis=1)


# Training the model

# In[ ]:


data.info()


# In[ ]:


x = data.drop(['depressed'], axis=1)
y = data.depressed


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:


rf = LogisticRegression(C = 1000.0,random_state = 0)
rf.fit(X_train_std, y_train)


# In[ ]:


#rf.fit(x,y)


# In[ ]:


from sklearn.metrics import mean_absolute_error
pred_train = rf.predict(X_train_std)
print (mean_absolute_error(pred_train,y_train))


# test

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


import seaborn as sns
sns.heatmap(test.isnull(),cbar=False, cmap='viridis')


# In[ ]:


test = test.dropna(axis=1)


# In[ ]:


test.head()


# In[ ]:


sns.heatmap(test.isnull(),cbar=False, cmap='viridis')


# In[ ]:


test['survey_date'] = pd.to_datetime(test['survey_date'])
test['month'] = test['survey_date'].dt.strftime('%m')
test['year'] = test['survey_date'].dt.strftime('%Y')
test['day'] = test['survey_date'].dt.strftime('%w')


# In[ ]:


test['year'] = pd.get_dummies(test['year'])


# In[ ]:


test = test.drop(s, axis=True)


# In[ ]:


test.info()


# In[ ]:


test.columns


# In[ ]:


q = ['fs_adskipm_often', 'asset_niceroof', 'cons_allfood',
       'cons_ed', 'med_vacc_newborns','ent_nonagbusiness',
       'cons_other','early_survey']
test = test.drop(q, axis=True)


# In[ ]:


x_test = test.drop(['surveyid'], axis=1)
test_pred = rf.predict(x_test)


# In[ ]:


from sklearn.model_selection import train_test_split
X_trains, X_tests, y_trains, y_tests = train_test_split(x_test,test_pred, test_size=0.33)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_trains)
X_train_st = sc.transform(X_trains)
X_test_st = sc.transform(X_tests)


# In[ ]:


rl = LogisticRegression(C = 1000.0,random_state = 0)
rl.fit(X_train_st, y_trains)


# In[ ]:


from sklearn.metrics import mean_absolute_error
pred_trains = rl.predict(X_train_st)
print (mean_absolute_error(pred_trains,y_trains))


# In[ ]:


q = {'surveyid': test["surveyid"], 'depressed': test_pred}
pred = pd.DataFrame(data=q)
pred = pred[['surveyid','depressed']]


# In[ ]:


pred.head


# In[ ]:


pred.to_csv('pred_set13.csv', index=False) #save to csv file


# In[ ]:




