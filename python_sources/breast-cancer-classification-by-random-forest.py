#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


dataset=pd.read_csv("../input/data.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


dataset.isnull().values.any()


# In[9]:


#the last column has null values. remove it.


# In[10]:


X=dataset.iloc[:,2:-1].values


# In[11]:


y=dataset.iloc[:,1].values


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


lb=LabelEncoder()


# In[14]:


y=lb.fit_transform(y)


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[18]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


sc=StandardScaler()


# In[21]:


p=pd.DataFrame(X_train)


# In[22]:


p.iloc[:,[29]].isnull().values.any()


# In[23]:


X_train=sc.fit_transform(X_train)


# In[24]:


X_test=sc.transform(X_test)


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


rf=RandomForestClassifier(n_estimators=28)


# In[54]:


rf.fit(X_train,y_train)


# In[55]:


y_pred=rf.predict(X_test)


# In[56]:


print('Accuracy is ',rf.score(X_test,y_test))


# In[58]:


y_pred


# In[59]:


y_test


# In[60]:


from sklearn.metrics import confusion_matrix


# In[61]:


cm=confusion_matrix(y_test,y_pred)


# In[62]:


cm


# In[63]:


from sklearn.metrics import precision_recall_fscore_support as score


# In[64]:


precision, recall, fscore, support = score(y_test, y_pred)


# In[65]:


print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[ ]:




