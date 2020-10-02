#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


#data.drop(columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"], inplace = True)
data.Sex[data.Sex == 'male'] = 0
data.Sex[data.Sex == 'female'] = 1


# In[ ]:


vec = data.groupby("Embarked").count()['PassengerId']


# Embarked
# 
# C    168
# 
# Q     77
# 
# S    644
# 
# Name: PassengerId, dtype: int64[](http://)

# In[ ]:


data.Embarked[data.Embarked == 'C'] = 1/vec[0]
data.Embarked[data.Embarked == 'Q'] = 1/vec[1]
data.Embarked[data.Embarked == 'S'] = 1/vec[2]


# In[ ]:


data.head()


# In[ ]:


data.drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis= 1, inplace = True) 


# In[ ]:


data = data.fillna(0)


# In[ ]:


train_o = data.Survived


# In[ ]:


train_i = data.drop(['Survived'], axis= 1) 


# In[ ]:


train_i.head()


# In[ ]:


train_o.head()


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(train_i,train_o)


# In[ ]:


data1 = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


data1.Sex[data1.Sex == 'male'] = 0
data1.Sex[data1.Sex == 'female'] = 1


# In[ ]:


vec = data1.groupby("Embarked").count()['PassengerId']


# In[ ]:


data1.Embarked[data1.Embarked == 'C'] = 1/vec[0]
data1.Embarked[data1.Embarked == 'Q'] = 1/vec[1]
data1.Embarked[data1.Embarked == 'S'] = 1/vec[2]


# In[ ]:


data1.head()


# In[ ]:


data1 = data1.fillna(0)
p_id=pd.DataFrame(data1['PassengerId'])
data1.drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis= 1, inplace = True) 


# In[ ]:


test_i = data1


# In[ ]:


p = model.predict(test_i)


# In[ ]:


submission = pd.DataFrame({'PassengerId':p_id['PassengerId'],'Survived':p})


# In[ ]:


submission.head()


# In[ ]:


filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




