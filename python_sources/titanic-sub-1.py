#!/usr/bin/env python
# coding: utf-8

# In[23]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np

from sklearn.ensemble import RandomForestClassifier


# In[2]:


train_df=pd.read_csv("../input/train.csv")


# In[3]:


train_df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[4]:


train_df.head()


# In[5]:


train_df['Embarked']= train_df.groupby('Embarked').ngroup()


# In[6]:


train_df['Sex']=train_df.groupby('Sex').ngroup()


# In[7]:


train_df.head()


# In[8]:


train_df.fillna(0,inplace=True)


# In[9]:


train_df_o=pd.DataFrame(train_df['Survived'])
# train_df['Survived']=train_df_o
train_df_i=train_df.drop(columns=['Survived'],inplace=False)


# In[49]:


model = RandomForestClassifier()


# In[51]:


model.fit(train_df_i,train_df_o)


# In[33]:


test_df=pd.read_csv("../input/test.csv")


# In[34]:


p_id=pd.DataFrame(test_df['PassengerId'])


# In[35]:


test_df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[36]:


test_df['Embarked']=test_df.groupby('Embarked').ngroup()
test_df['Sex']=test_df.groupby('Embarked').ngroup()


# In[37]:


test_df.head()


# In[38]:


test_df.fillna(0,inplace=True)


# In[52]:


p=model.predict(test_df)


# In[53]:


submission = pd.DataFrame({'PassengerId':p_id['PassengerId'],'Survived':p})


# In[54]:


submission.head()


# In[55]:


filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

