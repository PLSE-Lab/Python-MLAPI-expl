#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


features = pd.read_csv('../input/features.txt',header=None,sep=' ',names=('ID','Activity'))
labels = pd.read_csv('../input/activity_labels.txt',header=None,sep=' ',names=('ID','Sensor'))

X_train = pd.read_table('../input/X_train.txt',header=None,sep='\s+')
y_train = pd.read_table('../input/y_train.txt',header=None,sep='\s+')

X_test = pd.read_table('../input/X_test.txt',header=None,sep='\s+')
y_test = pd.read_table('../input/y_test.txt',header=None,sep='\s+')

train_sub = pd.read_table('../input/subject_train.txt',header=None,names=['SubjectID'])
test_sub = pd.read_table('../input/subject_test.txt',header=None,names=['SubjectID'])


# In[4]:


X_train.head()


# In[5]:


X_train.columns = features.iloc[:,1]
X_test.columns = features.iloc[:,1]


# In[6]:


y_train.columns = ['Activity']
y_test.columns = ['Activity']


# In[7]:


X_train['SubjectID'] = train_sub
X_test['SubjectID'] = test_sub


# In[8]:


X_train.head()


# In[9]:


y_train.head()


# In[10]:


X_train.isnull().sum().max()


# In[11]:


X_test.isnull().sum().max()


# In[12]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error


# In[13]:


from sklearn.model_selection import KFold,StratifiedKFold 


# In[14]:


from sklearn.model_selection import RepeatedKFold


# In[15]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[16]:


dtree = DecisionTreeClassifier()


# In[17]:


dtree.fit(X_train,y_train)


# In[18]:


y_pred = dtree.predict(X_test)


# In[19]:


print(rmsle(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[23]:


random_forest = RandomForestClassifier(n_estimators=6)


# In[24]:


#kfold = KFold().split(X_train,y_train)


# In[25]:


#for k, train in enumerate(kfold):
#    random_forest.fit(X_train[train], y_train[train])


# In[26]:


X_train.head()


# In[27]:


#Rp_Kfold = RepeatedKFold(n_splits=5).split(X_train,y_train)

#for k, train in enumerate(Rp_Kfold):
#    random_forest.fit(X_train[train],y_train[train])


# In[28]:


#St_Kfold = StratifiedKFold(n_splits=5).split(X_train,y_train)

#for k, train in enumerate(St_Kfold):
#    random_forest.fit(X_train[train],y_train[train])


# In[29]:


random_forest.fit(X_train,y_train)


# In[30]:


y_pred = random_forest.predict(X_test)


# In[31]:


print(rmsle(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[32]:


print('Using KFold cross validation technique')
print(rmsle(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[33]:


gra_boost = GradientBoostingClassifier()


# In[34]:


gra_boost.fit(X_train,y_train)
pred = gra_boost.predict(X_test)


# In[35]:


print(rmsle(y_test,pred))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


# In[ ]:




