#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
both = [train_data, test_data]


# In[ ]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train_data.columns[train_data.isnull().any()]


# In[ ]:


corrmat = train_data.corr()
matrix = np.triu(corrmat)
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corrmat, vmax=1.0,annot=True,fmt='.2g',cbar=False,mask=matrix, square=True)


# ### We can see clearly that SibSp and Parch column have strong correlation. And Pclass and Fare , Pclass and Age, Age and SibSp columns have strong negative correlation

# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', height=2.2, aspect=1.7)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5,color='r', ci=50)
grid.add_legend()


# In[ ]:


grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2, aspect=1.5)
grid.map(plt.hist, 'Age',color='g', alpha=.7, bins=20)
grid.add_legend();


# In[ ]:


for dataset in both:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# In[ ]:


for dataset in both:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'VIP')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'Survived']].groupby(['Title']).mean()


# #### According to the survival rate according to the titles , We will convert this categorical values to numerical values.

# In[ ]:


for data in both:
    data.Title = data.Title.replace(to_replace=['Master','Miss','Mr','Mrs','VIP'], value=[0.575,0.702,0.156,0.793,0.347])


# In[ ]:


train_data.head()


# ### Now We can remove the Name column

# In[ ]:


for data in both:
    data.drop('Name', axis=1, inplace=True)


# In[ ]:


train_data.head()


# ## Since Cabin have lot of null values , We are going to drop it.

# In[ ]:


train_data['Cabin'].describe()


# In[ ]:


train_data.info()


# In[ ]:


train_data.drop('Cabin',axis=1, inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


ID = np.array(test_data.PassengerId)
train_data.drop('PassengerId', axis=1, inplace=True)
test_data.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


train_data.columns[train_data.isnull().any()]


# In[ ]:


test_data.columns[test_data.isnull().any()]


# In[ ]:


train_data['Age'].mode()


# In[ ]:


train_data.head()


# ## I don't think the Ticket column will have much effect on the Survival predictions. So We are going to drop this column.

# In[ ]:


train_data.drop('Ticket', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


train_data['Age'].describe()


# In[ ]:


train_data['Age'] = train_data['Age'].fillna(np.mean(train_data['Age']))


# In[ ]:


train_data.columns[train_data.isnull().any()]


# In[ ]:


train_data[train_data['Embarked'].isnull()]


# In[ ]:


train_data['Embarked'].value_counts()


# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna('S')


# In[ ]:


train_data.columns[train_data.isnull().any()]


# In[ ]:


test_data['Age'] = test_data['Age'].fillna(np.mean(test_data['Age']))


# In[ ]:


test_data.columns[test_data.isnull().any()]


# In[ ]:


test_data[test_data.Fare.isnull()]


# In[ ]:


test_data.Fare.describe()


# In[ ]:


test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())


# In[ ]:


train_data.columns


# In[ ]:


test_data.columns


# In[ ]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[ ]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[ ]:


print(train_data.shape, test_data.shape)


# In[ ]:


train_data = train_data.replace(to_replace=['female','male'],value=[1,0])
test_data = test_data.replace(to_replace=['female','male'],value=[1,0])


# In[ ]:


pd.crosstab(train_data['Embarked'], train_data['Survived'])


# In[ ]:


train_data.Embarked= train_data.Embarked.replace(to_replace=['C','S','Q'], value=[1.24,0.51,0.63])
test_data.Embarked = test_data.Embarked.replace(to_replace=['C','S','Q'], value=[1.24,0.51,0.63])


# In[ ]:


#num_features = [cat for cat in train_data.columns if train_data[cat].dtype in ['float64','int64']]
#cat_features = [cat for cat in train_data.columns if train_data[cat].dtype not in ['float64','int64']]
#print(cat_features,'\n',num_features)


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.shape


# In[ ]:


valid_data = train_data[int(0.8 * train_data.shape[0]):]
t_data = train_data[:int(0.8 * train_data.shape[0])]


# In[ ]:


t_data.info()


# In[ ]:


train_data.info()


# In[ ]:


import lightgbm as lgb

feature_cols = train_data.columns.drop('Survived')
dtrain = lgb.Dataset(t_data[feature_cols], label=t_data['Survived'])
dvalid = lgb.Dataset(valid_data[feature_cols], label=valid_data['Survived'])

param = {'num_leaves':64 , 'objective':'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets = [dvalid], early_stopping_rounds=10,verbose_eval=False)


# In[ ]:


y_pred = bst.predict(test_data[feature_cols])


# In[ ]:


model = lgb.LGBMClassifier(learning_rate=0.05, num_leaves=61,n_estimators=1000)
model.fit(train_data[feature_cols], train_data['Survived'])


# In[ ]:


y_pred1 = model.predict(test_data)


# In[ ]:


x = [int(i+0.65) for i in y_pred]
x[:3]


# In[ ]:


submit = pd.DataFrame({'PassengerId':ID, 'Survived':y_pred1})


# In[ ]:


submit.to_csv('submission.csv', index=False)


# In[ ]:




