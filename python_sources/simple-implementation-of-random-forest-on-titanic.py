#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 


# In[ ]:


data=pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


features=['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']


# In[ ]:


data=data[features]
data.head()


# In[ ]:


data.info()


# In[ ]:


data.fillna(data.mean(),inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['Cabin'].fillna(0,inplace=True)


# In[ ]:


y=data


# In[ ]:


j=0
for i in data['Cabin']:
    if(i!=0):
        data['Cabin'][j]=1
    j=j+1


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data=pd.get_dummies(data)


# In[ ]:


data.head()


# In[ ]:


data.drop(['Sex_female','Cabin_1'],axis=1,inplace=True)


# In[ ]:


X=data.drop(['Survived'],axis=1)


# In[ ]:


y=data['Survived']


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.33, random_state=42)


# In[ ]:


train_X.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model=RandomForestClassifier()
model.fit(train_X,train_y)


# In[ ]:


pred=model.predict(test_X)


# In[ ]:


from sklearn import metrics


# In[ ]:


score=metrics.accuracy_score(test_y,pred)


# In[ ]:


score


# In[ ]:




