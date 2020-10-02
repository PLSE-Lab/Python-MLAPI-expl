#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


data=pd.read_csv('../input/500_Person_Gender_Height_Weight_Index.csv')


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data.Gender.describe()


# In[8]:


X=data.drop(['Index'],axis=1)
X=pd.get_dummies(X)
y=data['Index']


# In[9]:


X.head()


# In[10]:


y.head()


# In[11]:


X.dtypes


# In[12]:


import sklearn.model_selection as model_selection
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=100)


# In[13]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
for i in range(10,100,10):
    reg=RandomForestClassifier(n_estimators=i,max_depth=5,max_features='sqrt',oob_score=True,random_state=100)
    reg.fit(X_train,y_train)
    oob=reg.oob_score_
    print('For n_estimators = '+str(i))
    print('OOB score is '+str(oob))
    print('************************')


# In[15]:


reg=RandomForestClassifier(n_estimators=50,max_depth=5,max_features='sqrt',oob_score=True,random_state=100)
reg.fit(X_train,y_train)


# In[16]:


reg.score(X_test,y_test)


# In[17]:


reg.oob_score_


# In[18]:


feature_importance=reg.feature_importances_


# In[19]:


feat_imp=pd.Series(feature_importance,index=X.columns.tolist())


# In[20]:


feat_imp.sort_values(ascending=False)


# In[21]:


pred=reg.predict(X_test)


# In[ ]:





# In[ ]:




