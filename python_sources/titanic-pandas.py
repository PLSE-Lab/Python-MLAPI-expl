#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings(action="ignore")

# Any results you write to the current directory are saved as output.


# In[ ]:


tra=pd.read_csv('../input/train.csv')
tes=pd.read_csv('../input/test.csv')


# In[ ]:


x=tra.drop(['Name','PassengerId','Ticket','Survived'],axis=1)
x_t=tes.drop(['Name','PassengerId','Ticket'],axis=1)


# In[ ]:


x.isna().sum()


# In[ ]:


x_t.isna().sum()


# In[ ]:


x.Age=x.Age.fillna(x.Age.mean())
x_t.Age=x_t.Age.fillna(x_t.Age.mean())


# In[ ]:


x.Cabin=x.Cabin.fillna('U')
x_t.Cabin=x_t.Cabin.fillna('U')


# In[ ]:


x.Embarked=x.Embarked.fillna('S')
x_t.Embarked=x_t.Embarked.fillna('S')


# In[ ]:


x_t.Fare=x_t.Fare.fillna(x_t.Fare.mean())


# In[ ]:


x.Cabin = x.Cabin.map(lambda z: z[0])
x_t.Cabin = x_t.Cabin.map(lambda z: z[0])


# In[ ]:


#x['Total Family']=x['SibSp']+x['Parch']
#x_t['Total Family']=x_t['SibSp']+x['Parch']


# In[ ]:


#x=x.drop(['Parch','SibSp'],axis=1)
#x_t=x_t.drop(['Parch','SibSp'],axis=1)


# In[ ]:


x


# In[ ]:


x_t


# In[ ]:


#x['Cabin'] = pd.Categorical(x['Cabin'])


# In[ ]:


#x['Embarked']=pd.Categorical(x['Embarked'])


# In[ ]:


x= pd.get_dummies(x)
x_t=pd.get_dummies(x_t)


# In[ ]:


x.head()


# In[ ]:


x_t.head()


# In[ ]:


x=x.drop(['Cabin_T'],axis=1)


# In[ ]:


y=tra['Survived']


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier(n_estimators=100000,random_state=0)
reg.fit(x_train,y_train)


# In[ ]:


#from sklearn.ensemble import GradientBoostingClassifier
#reg = GradientBoostingClassifier()
#reg.fit(x_train,y_train)


# In[ ]:


y_pred=reg.predict(x_val)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_pred)


# In[ ]:


pred=reg.predict(x_t)
new_pred=pred.astype(int)
output=pd.DataFrame({'PassengerId':tes['PassengerId'],'Survived':new_pred})
output.to_csv('Titanic.csv', index=False)

