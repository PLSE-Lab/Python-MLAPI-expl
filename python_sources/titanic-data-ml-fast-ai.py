#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from fastai import *
from fastai.vision import *
from fastai.tabular import *


# In[3]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[4]:


test = pd.read_csv("../input/test.csv")
test.isnull().sum()


# In[5]:


train["Embarked"] = train["Embarked"].fillna("C")
test["Embarked"] = test["Embarked"].fillna("C")


# In[6]:


train['Fare'].fillna(train['Fare'].median(), inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)


# In[7]:


train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)


# In[8]:


train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(train_title)
train["Title"].head()


# In[9]:


test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
test["Title"] = pd.Series(test_title)
test["Title"].head()


# In[10]:


train_group = train.groupby(['Sex','Pclass', 'Title']) 
train_group.Age.median()
train.Age = train_group.Age.apply(lambda x: x.fillna(x.median()))
test_group = test.groupby(['Sex','Pclass', 'Title'])  
test_group.Age.median()
train.Age = test_group.Age.apply(lambda x: x.fillna(x.median()))


# In[11]:


test_id = test['PassengerId'] 
var = 'Survived'
category_names = ['Title', 'Sex', 'Ticket', 'Cabin', 'Embarked']
continues_names = [ 'Age', 'SibSp', 'Parch', 'Fare']
print("Categorical columns are : ", category_names)
print('Continuous numerical columns are :', continues_names)
procs = [FillMissing, Categorify, Normalize]


# In[12]:


test = TabularList.from_df(test, cat_names=category_names, cont_names=continues_names, procs=procs)


# In[15]:


data = (TabularList.from_df(train, path='.', cat_names=category_names, cont_names=continues_names, procs=procs)
                        .split_by_idx(list(range(0,200)))
                        .label_from_df(cols = var)
                        .add_test(test, label=0)
                        .databunch())


# In[16]:


data.show_batch(rows=10)


# In[17]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy, emb_drop=0.1)


# In[18]:


learn.lr_find()


# In[19]:


learn.recorder.plot()


# In[28]:


learn.fit(40, slice(1e-01))


# In[29]:


preds, targets = learn.get_preds()


# In[30]:


predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)


# In[31]:


predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)


# In[32]:


submission = pd.DataFrame({'PassengerId': test_id, 'Survived': labels})
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




