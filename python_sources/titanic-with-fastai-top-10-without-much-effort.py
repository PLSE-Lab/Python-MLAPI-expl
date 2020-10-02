#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


import seaborn as sns


# In[ ]:


train_test_df = [test, train]

for data in train_test_df:
    data['Tittle'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:



train["Age"].fillna(train.groupby("Tittle")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Tittle")["Age"].transform("median"), inplace=True)


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:




train.drop(['Name', 'Ticket',], axis =1, inplace=True)


# In[ ]:


train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train['Age'].dtypes


# In[ ]:


test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


test.drop(['Name', 'Ticket', ], axis =1, inplace=True)


# In[ ]:


train['Embarked'].fillna('S',inplace=True)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()


# In[ ]:


test.fillna(method='ffill',inplace=True)


# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


test['Embarked'].dtypes


# In[ ]:


from fastai.tabular import *


# In[ ]:





# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


dep_var='Survived'
cat_names=['Sex','Embarked','Pclass','Tittle']
cont_names = [ 'Age','SibSp','Parch','Fare']
procs = [FillMissing, Categorify,Normalize]


# In[ ]:


test1 = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)


# In[ ]:


data = (TabularList.from_df(train,path='.', cont_names = cont_names,cat_names=cat_names, procs=procs)
                           .split_by_idx(list(range(600,800)))
                           .label_from_df(cols=dep_var)
                           .add_test(test1,label=0)
                           .databunch())


# In[ ]:



data.show_batch(rows=10)


# In[ ]:


np.random.seed(101)
learn = tabular_learner(data, layers=[30,10], metrics=accuracy)


# In[ ]:





# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit(10,1e-02)


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit(10, 1e-06)


# In[ ]:


learn.show_results(20)


# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)


# In[ ]:


sub_df = pd.DataFrame({
    'PassengerId': test.PassengerId,
    'Survived':labels
})
sub_df.to_csv('submission22.csv', index=False)


# In[ ]:


sub_df.head()


# In[ ]:




