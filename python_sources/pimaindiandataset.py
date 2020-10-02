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


df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


X = df


# In[ ]:


y = df.Outcome


# In[ ]:


X.drop("Outcome",axis = 1,inplace = True)


# In[ ]:


X


# In[ ]:


y


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X,X_test,y,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)


# In[ ]:


lr.fit(X,y)


# In[ ]:


pred = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(pred,y_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(pred,y_test)*100


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, random_state = 1)
rf.fit(X,y)


# In[ ]:


pred = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,pred)*100


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier(random_state = 1,n_estimators = 100)


# In[ ]:


xgb.fit(X,y)


# In[ ]:


pred = xgb.predict(X_test)


# In[ ]:


accuracy_score(y_test,pred)*100


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


model = KNeighborsClassifier(n_neighbors = 50)


# In[ ]:


model.fit(X,y)


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(pred,y_test)*100


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model2 = SVC()


# In[ ]:


model2.fit(X,y)


# In[ ]:


pred3 = model2.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred3)


# In[ ]:


X.columns


# In[ ]:


X.corr()


# In[ ]:


df.columns


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


sc.fit_transform(X)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 1, n_estimators = 1000)
rfc.fit(X,y)


# In[ ]:


pred5 = rfc.predict(X_test)


# In[ ]:


accuracy_score(y_test,pred5)*100


# In[ ]:




