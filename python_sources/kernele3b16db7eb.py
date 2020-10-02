#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


df_train = pd.read_csv('../input/mnist_train.csv')
df_test = pd.read_csv('../input/mnist_test.csv')


# In[ ]:


X_train = df_train.iloc[:, 1:785]
y_train = df_train.iloc[:, 0]


# In[ ]:


X_test = df_test.iloc[:, 1:785]
y_test=df_test.iloc[:, 0]


# In[ ]:


y_test


# In[ ]:


#RFT

from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
clf1.fit(X_train,y_train)
clf1.score(X_test,y_test)


# In[ ]:


y_pred1=clf1.predict(X_test)


# In[ ]:


y_pred1


# cm1=confusion_matrix(y_test,y_pred1)
# acc1=accuracy_score(y_test,y_pred1)
# from sklearn.model_selection import KFold, cross_val_score
# models=[]
# results=[]
# names=[]
# models.append(('RF',RandomForestClassifier()))
# for name,model in models:
#     kfold=KFold(n_splits=10)
#     cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='precision_macro')
#     results.append(cv_results)
#     names.append(name)
#     msg="%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
#     print(msg)

# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(X_train, y_train)
model.score(X_test,y_test)


# In[ ]:


import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)

