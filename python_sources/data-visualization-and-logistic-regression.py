#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data =pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# ## Analysing Data

# In[ ]:


sns.countplot(x="Survived",data=data)


# #### Here Label 0 Denotes people who did not survive and 1 People who survive so we can see that more than 500 people died and more than 300 people survived. Lets divide this plot according to sex.

# In[ ]:


sns.countplot(x="Survived",hue="Sex",data=data)


# #### We can see that women had more probablity of surviving this than men. Lets check for passanger class next

# In[ ]:


sns.countplot(x="Survived",hue="Pclass",data=data)


# In[ ]:


data["Age"].plot.hist()


# # Data Wrangling and Cleaning

# In[ ]:


data.isnull()[:10] ## Returns true if the value is NULL otherwise false


# In[ ]:


data.isnull().sum() 


# #### Here we can see that for cabin column 687 values are NULL so we should drop it. 

# In[ ]:


data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


data.dropna(inplace=True)#drop every row which has null value


# In[ ]:


data.head()


# In[ ]:


sex=pd.get_dummies(data['Sex'],drop_first=True)
sex.head()


# In[ ]:


embark=pd.get_dummies(data['Embarked'],drop_first=True)
embark.head()


# In[ ]:


pcl=pd.get_dummies(data['Pclass'],drop_first=True)
pcl.head()


# In[ ]:


data=pd.concat([data,sex,pcl,embark],axis=1)
data.head()


# In[ ]:


data=data.drop(['Pclass','Sex','Embarked','Name','Ticket','PassengerId'],axis=1)
data.head()


# # Training

# In[ ]:


x=data.drop('Survived',axis=1)
y=data['Survived']


# In[ ]:


print(x.head())


# In[ ]:


print(y.head())


# In[ ]:


import sklearn
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x,y)


# In[ ]:


test=pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


passengerid=test['PassengerId'].values
passengerid[:10]


# In[ ]:


sex=pd.get_dummies(test['Sex'],drop_first=True)
sex.head()


# In[ ]:


embark=pd.get_dummies(test['Embarked'],drop_first=True)
embark.head()


# In[ ]:


pcl=pd.get_dummies(test['Pclass'],drop_first=True)
pcl.head()


# In[ ]:


test=pd.concat([test,sex,pcl,embark],axis=1)
test.head()


# In[ ]:


test=test.drop(['Pclass','Sex','Embarked','Name','Ticket','PassengerId','Cabin'],axis=1)
test.head()


# In[ ]:


test.fillna(value=0,inplace=True)#fill values
test.head()


# In[ ]:


predictions=logmodel.predict(test)


# In[ ]:


predictions[:10]


# In[ ]:


predictions=predictions.tolist()
predictions[:10]


# In[ ]:


passengerid=passengerid.tolist()
passengerid[:10]


# In[ ]:


len(predictions)


# In[ ]:


submission={
    'PassengerId':passengerid,
    'Survived':predictions
}


# In[ ]:



df = pd.DataFrame(submission)


# In[ ]:


df.head()


# In[ ]:


df.to_csv('submission.csv',index=False)


# In[ ]:




