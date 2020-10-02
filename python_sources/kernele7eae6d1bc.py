#!/usr/bin/env python
# coding: utf-8

# In[32]:


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


# In[33]:



import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


train = pd.read_csv("../input/train.csv")


# In[35]:


train.head()


# In[36]:


train.info()


# In[37]:


train.describe()


# misising data

# In[38]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[39]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[40]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[41]:


sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[42]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[43]:


sns.countplot(x='SibSp',data=train)


# In[44]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[45]:


import cufflinks as cf # just to look
cf.go_offline()


# In[46]:


train['Fare'].iplot(kind='hist',bins=30,color='green')


# Data Cleaning
# 
# We want to fill in missing age 

# In[47]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# passengers in the higher classes tend to be older

# In[48]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[49]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# check that heat map again

# In[50]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# drop the Cabin column

# In[51]:


train.drop('Cabin',axis=1,inplace=True)


# In[52]:


train.head()


# In[53]:


train.dropna(inplace=True)


# In[54]:


train.info()


# use dummy variables instead of sex and embarked

# In[55]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[56]:


train.head()


# In[57]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[58]:


train = pd.concat([train,sex,embark],axis=1)


# In[59]:


train.head()


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[64]:


predictions = logmodel.predict(X_test)


# Evaluation

# In[66]:


from sklearn.metrics import classification_report


# In[67]:


print(classification_report(y_test,predictions))

