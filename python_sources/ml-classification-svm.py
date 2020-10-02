#!/usr/bin/env python
# coding: utf-8

# # 86% accuracy by a very simple approach !! 
# * Standard Scaling
# * SVC
# * GridSearchCV
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm


# In[ ]:


data=pd.read_csv("../input/heart-disease-uci/heart.csv")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.isnull().any()


# In[ ]:


data['target'].value_counts()


# In[ ]:


data.corr()


# In[ ]:


Y_train=data['target']
X_train=data.drop('target',axis=1)


# In[ ]:


data.corr()


# In[ ]:


arr=np.array(data.columns)
arr[:-1]


# In[ ]:


# X_data = data.drop('target',axis=1)
# scaler = RobustScaler()
# scaler = scaler.fit(X_data)
# X_scaled_data = scaler.transform(X_data)
# X_data = pd.DataFrame(X_scaled_data,columns=X_data.columns)


# In[ ]:


rs = StandardScaler()
train_rs=rs.fit_transform(X_train)
print(train_rs.shape)
train_scaled=pd.DataFrame(train_rs,columns=X_train.columns)
train_scaled.head()
print(train_scaled.shape)


# In[ ]:


Y_train.shape
train_scaled.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_scaled, Y_train, test_size=0.20)


# In[ ]:


clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
confusion_matrix(y_pred,y_test)
print(classification_report(y_pred,y_test))


# # GridSearchCV

# In[ ]:


from sklearn.model_selection import ShuffleSplit,GridSearchCV
from sklearn.metrics import f1_score,make_scorer


# In[ ]:


params={
    'gamma':[0.1,0.01,0,0.001,0.0001],
    'C':[0.1,10,100,1000],
    'kernel':['rbf']
}
grid=GridSearchCV(clf,param_grid=params,scoring=make_scorer(f1_score))


# In[ ]:


grid_clf=grid.fit(X_train,y_train)


# In[ ]:


grid_clf.best_estimator_


# In[ ]:


grid_best=grid_clf.best_estimator_


# In[ ]:


y_grid_pred=grid_best.predict(X_test)


# In[ ]:


print(confusion_matrix(y_grid_pred,y_test))
print(classification_report(y_grid_pred,y_test))


# ## I am working on this dataset so I will keep updating it. <br><br>Consider an Upvote if you like it !

# In[ ]:




