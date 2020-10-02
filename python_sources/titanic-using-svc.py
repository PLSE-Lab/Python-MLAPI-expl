#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


data=train.loc[:,['Pclass','Sex','Age','SibSp','Parch']]
target=train.loc[:,'Survived']


# In[ ]:


data.head()


# In[ ]:


data['Sex'].unique()


# In[ ]:


data.head()


# In[ ]:


sexd=data.loc[:,'Sex']
sexd=pd.get_dummies(sexd)


# In[ ]:


sexd.head()


# In[ ]:


data=data.join(sexd)


# In[ ]:


data.drop(columns='Sex',axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


testdata=test.loc[:,['Pclass','Sex','Age','SibSp','Parch']]
testdata.head()


# In[ ]:


sexdt=testdata.loc[:,'Sex']
sexdt=pd.get_dummies(sexdt)


# In[ ]:


sexdt.head()


# In[ ]:


testdata.drop(columns='Sex',inplace=True)
testdata=testdata.join(sexdt)
testdata.head()


# In[ ]:


from sklearn.preprocessing import Imputer

trainData = Imputer().fit_transform(data)
testData=Imputer().fit_transform(testdata)


# In[ ]:


from sklearn.svm import SVC
titanic=SVC()


# In[ ]:


titanic


# In[ ]:


titanic.fit(trainData,target)


# In[ ]:


pred=titanic.predict(trainData)


# In[ ]:


from sklearn.metrics import classification_report as clf
print(clf(target,pred))


# In[ ]:


predict=titanic.predict(testData)


# In[ ]:


print(predict)


# In[ ]:


id=test.loc[:,'PassengerId']


# In[ ]:


id.shape


# In[ ]:


result=pd.DataFrame(id)
result.shape


# In[ ]:


predict.shape


# In[ ]:


result['Survived']=predict


# In[ ]:


result.head()


# In[ ]:





# In[ ]:




