#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Load Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col = "PassengerId")
train.head()
test = pd.read_csv('../input/test.csv', index_col = "PassengerId")
test.head()


# ## Explore

# ### Sex

# In[ ]:


sns.countplot(data = train, x = 'Sex', hue = 'Survived')


# In[ ]:


pd.pivot_table(train, index = 'Sex', values = 'Survived')    


# ### Pclass

# In[ ]:


sns.countplot(data = train, x = 'Pclass', hue = 'Survived')


# In[ ]:


pd.pivot_table(train, index = 'Pclass', values = 'Survived')


# ### Embarked

# In[ ]:


sns.countplot(data = train, x = 'Embarked', hue = 'Survived')


# In[ ]:


pd.pivot_table(train, index = 'Embarked', values = 'Survived')


# ### Age & Fare

# In[ ]:


sns.lmplot(data = train, x = 'Age',  y='Fare', hue = 'Survived', fit_reg = False)


# In[ ]:


low_Fare = train[train['Fare'] < 100]


# In[ ]:


sns.lmplot(data = low_Fare, x = 'Age', y = 'Fare', hue = 'Survived', fit_reg = False)


# ### Family Size

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


train[['SibSp', 'Parch', 'FamilySize']].head()


# In[ ]:


sns.countplot(data = train, x = 'FamilySize', hue = 'Survived')


# In[ ]:


train.loc[train['FamilySize'] == 1, 'FamilyType'] = 'Single'
train.loc[(train['FamilySize'] > 1) & (train['FamilySize'] < 5), 'FamilyType'] = 'Nuclear'
train.loc[train['FamilySize'] >= 5, 'FamilyType'] = 'Big'

train[['FamilySize', 'FamilyType']].head()


# In[ ]:


sns.countplot(data = train, x = 'FamilyType', hue = 'Survived')


# In[ ]:


pd.pivot_table(train, index = 'FamilyType', values = 'Survived')


# ### Name

# In[ ]:


train['Name'].head()


# In[ ]:


def get_title(name):
    return name.split(', ')[1].split('. ')[0]

train['Name'].apply(get_title).unique()


# In[ ]:


train.loc[train['Name'].str.contains('Mr'), 'Title'] = 'Mr'
train.loc[train['Name'].str.contains('Mrs'), 'Title'] = 'Mrs'
train.loc[train['Name'].str.contains('Miss'), 'Title'] = 'Miss'
train.loc[train['Name'].str.contains('Master'), 'Title'] = 'Master'

train[['Name','Title']].head()


# In[ ]:


sns.countplot(data = train, x = 'Title', hue = 'Survived')


# In[ ]:


pd.pivot_table(train, index = "Title", values = 'Survived')


# ## Preprocessing

# In[ ]:


train.loc[train['Sex'] == 'female', 'Sex_encode'] = 0 
train.loc[train['Sex'] == 'male', 'Sex_encode'] = 1
test.loc[test['Sex'] == 'female', 'Sex_encode'] = 0 
test.loc[test['Sex'] == 'male', 'Sex_encode'] = 1


# In[ ]:


train[['Sex', 'Sex_encode']].head()


# In[ ]:


test[['Sex', 'Sex_encode']].head()


# ### Fillin in missing Fare

# In[ ]:


train[train['Fare'].isnull()]


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


train['Fillin_Fare'] = train['Fare']


# In[ ]:


test['Fillin_Fare'] = test['Fare']


# In[ ]:


test.loc[test['Fare'].isnull(), 'Fillin_Fare']= 0
test.loc[test['Fare'].isnull(), ['Fare', 'Fillin_Fare']]


# ### Embarked

# In[ ]:


train['Embarked_C'] = train['Embarked'] == 'C'
train['Embarked_Q'] = train['Embarked'] == 'Q'
train['Embarked_S'] = train['Embarked'] == 'S'

train[['Embarked','Embarked_S', 'Embarked_Q', 'Embarked_C']].head()


# In[ ]:


test['Embarked_C'] = test['Embarked'] == 'C'
test['Embarked_Q'] = test['Embarked'] == 'Q'
test['Embarked_S'] = test['Embarked'] == 'S'

test[['Embarked','Embarked_S', 'Embarked_Q', 'Embarked_C']].head()


# ### Age

# In[ ]:


train['Child'] = train['Age'] < 15
train[['Age', 'Child']].head()


# In[ ]:


test['Child'] = test['Age'] < 15
test[['Age', 'Child']].head()


# ### FamilySize

# In[ ]:


train['Single'] = train['FamilySize'] == 1
train['Nuclear'] = (train['FamilySize'] > 1)  & (train['FamilySize'] < 5)
train['Big'] = train['FamilySize'] >= 5

train[['FamilySize', 'Single', 'Nuclear', 'Big']].head()


# In[ ]:


test['Single'] = test['FamilySize'] == 1
test['Nuclear'] = (test['FamilySize'] > 1)  & (test['FamilySize'] < 5)
test['Big'] = test['FamilySize'] >= 5

test[['FamilySize', 'Single', 'Nuclear', 'Big']].head()


# ### Name

# In[ ]:


train['Master'] = train['Name'].str.contains('Master')

train[['Name', 'Master']].head()


# In[ ]:


test['Master'] = test['Name'].str.contains('Master')

test[['Name', 'Master']].head()


# ### Train

# In[ ]:


feature_names = ['Pclass', 'Sex_encode', 'Fillin_Fare', 'Embarked_C','Embarked_Q', 'Embarked_S', 'Child', "Single", "Nuclear", 'Big', 'Master']
feature_names


# In[ ]:


label_name = 'Survived'


# In[ ]:


X_train = train[feature_names]
X_train.head()


# In[ ]:


X_test = test[feature_names]
X_test.head()


# In[ ]:


y_train = train['Survived']
y_train.head()


# ## Use Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = 8, random_state = 0)
model


# In[ ]:


model.fit(X_train, y_train)


# ## Visualize

# In[ ]:


import graphviz
from sklearn.tree import export_graphviz

dot_tree = export_graphviz(model, feature_names = feature_names, class_names=["Perish", "Survived"], out_file = None)
graphviz.Source(dot_tree)


# ## Predict

# In[ ]:


predictions = model.predict(X_test)
print(predictions.shape)
predictions[0:10]


# ## Submit

# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv', index_col = 'PassengerId')


# In[ ]:


submission['Survived'] = predictions
print(submission.shape)
submission.head()


# END... Thank you
