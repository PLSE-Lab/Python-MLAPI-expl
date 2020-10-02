#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset_train=pd.read_csv('../input/train.csv')


# In[3]:


dataset_test=pd.read_csv('../input/test.csv')


# In[4]:


dataset_train.head()


# In[5]:


dataset_test.head()


# In[6]:


dataset_train.isnull().values.any()


# In[7]:


dataset_test.isnull().values.any()


# In[8]:


dataset_train.info()


# In[9]:


dataset_test.info()


# In[10]:


dataset_train.describe()


# In[11]:


dataset_test.describe()


# In[12]:


dataset_train.shape


# In[13]:


dataset_test.shape


# In[14]:


dataset_train.head()


# In[15]:


X=dataset_train.iloc[:,1:-1].values


# In[16]:


X.shape


# In[17]:


y=dataset_train.iloc[:,-1].values


# In[18]:


y.shape


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)


# In[21]:


X_train.shape


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


sc=StandardScaler()


# In[24]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[26]:


rf=RandomForestClassifier(n_estimators=54,random_state=101,min_samples_split=4,criterion='entropy')


# In[27]:


rf.fit(X_train,y_train)


# In[28]:


y_pred=rf.predict(X_test)


# In[29]:


rf.score(X_test,y_test)


# In[30]:


dataset_test.head()


# In[31]:


X_new=dataset_test.iloc[:,1:].values


# In[32]:


X_new.shape


# In[33]:


X_new=sc.transform(X_new)


# In[34]:


y_new=rf.predict(X_new)


# In[35]:


y_new.shape


# In[36]:


upload=pd.DataFrame(y_new,dataset_test['Id'])


# In[37]:


upload.head()


# In[38]:


upload.to_csv('Submisssion_new.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




