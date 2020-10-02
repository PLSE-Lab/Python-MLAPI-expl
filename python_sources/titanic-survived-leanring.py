#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full = train.append(test,ignore_index=True)
full.shape


# In[ ]:


full.info()


# In[ ]:


full.Embarked.value_counts()


# In[ ]:


full.Age = full.Age.fillna(full.Age.mean())
full.Fare = full.Fare.fillna(full.Fare.mean())
full.Embarked = full.Embarked.fillna('S')


# In[ ]:


full.info()


# In[ ]:


full = full.drop('Cabin',axis=1)


# In[ ]:


full.info()


# In[ ]:


full.Sex = full.Sex.map({'male':1,'female':0})
full.head()


# In[ ]:


full.Embarked.head()


# In[ ]:


embarked = pd.DataFrame()
embarked = pd.get_dummies(full.Embarked,prefix='Embarked')
embarked.head()


# In[ ]:


pclass = pd.DataFrame()
pclass = pd.get_dummies(full.Pclass,prefix='Pclass')
pclass.head()


# In[ ]:


family = pd.DataFrame()
family['Family'] = full.Parch + full.SibSp + 1
family.head()


# In[ ]:


full.head()


# In[ ]:


full = pd.concat([full,embarked,pclass,family],axis=1)


# In[ ]:


full.head()


# In[ ]:


full = full.drop(['Embarked','Pclass','SibSp','Parch'],axis=1)


# In[ ]:


full.head()


# In[ ]:


corr = full.corr()


# In[ ]:


corr


# In[ ]:


corr.Survived.sort_values(ascending=False)


# In[ ]:


full.head()


# In[ ]:


full_X = pd.concat([full.Age,full.Sex,full.Fare,full.Family,embarked,pclass],axis=1)


# In[ ]:


full_X.shape


# In[ ]:


X = full_X.loc[0:890,:]
y = full.loc[0:890,'Survived']


# In[ ]:


X.shape


# In[ ]:


test_X = full_X.loc[891:,:]
test_X.shape


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


logreg.score(X_test,y_test)


# In[ ]:


test_X.head()


# In[ ]:


pred = logreg.predict(test_X)


# In[ ]:


pred = pred.astype(int)


# In[ ]:


ids = full.loc[891:,'PassengerId']


# In[ ]:


Pred = pd.DataFrame({'PassengerId':ids,'Survived':pred})


# In[ ]:


Pred.shape


# In[ ]:


Pred.head()


# In[ ]:


Pred.to_csv('titanic_pred.csv')


# In[ ]:




