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


# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set() #setting seaborn default for plots
#plt.style.use('seaborn')
#sns.set(font_scale=2)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# ## 1. EDA 
# 1)check data shape, info

# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.head(3)


# 2) data visualization 
# 
# bar plot
# -> norminal variable : Sex, Cabin, Embarked 
# 
# -> ordinal variable : Pclass
# 
# -> ratio variable : SibSp, Parch    

# In[ ]:


def bar_plot(feature, figsize=(10,3), legend=True):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived,dead],index=['Survived','dead'])
    print(df)
    df.plot(kind='barh',stacked=True, figsize=figsize, legend=legend)


# In[ ]:


bar_plot('Sex') 


# In[ ]:


bar_plot('Embarked') 


# In[ ]:


bar_plot('SibSp') 


# In[ ]:


bar_plot('Parch')


# In[ ]:


bar_plot('Cabin',legend=False) 


# In[ ]:


train.isnull().sum()  #cabin: too many null data


# In[ ]:


initial = train.Name.str.extract(r'([a-zA-Z]+)\.')
initial[0].unique()


# In[ ]:


train['Initial'] = train.Name.str.extract(r'([a-zA-Z]+)\.')
test['Initial'] = test.Name.str.extract(r'([a-zA-Z]+)\.')


# In[ ]:


pd.crosstab(train['Initial'], train['Sex']).T


# In[ ]:


pd.crosstab(test['Initial'], test['Sex']).T


# ## 2. Feature engineering
# 
# 1) Name -> Initial

# In[ ]:


bar_plot('Initial')


# In[ ]:


test.Initial.unique()


# In[ ]:


origin = ['Capt','Col','Countess','Don','Dr'
          ,'Jonkheer','Lady','Major','Master','Miss'
          ,'Mlle','Mme','Mr','Mrs','Ms'
          ,'Rev','Sir','Dona']
replace = ['etc','etc','etc','etc','Dr'
           ,'etc','etc','etc','Master','Miss'
           ,'etc','etc','Mr','Mrs','etc'
           ,'Rev','etc','etc']    
train.Initial.replace(origin,replace, inplace=True)
test.Initial.replace(origin, replace, inplace=True)
train.Initial.unique()
test.Initial.unique()


# In[ ]:


bar_plot('Initial')


# Processing null value 

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.Age = train.Age.fillna(train.Age.mean())
test.Age = test.Age.fillna(train.Age.mean())


# In[ ]:


train.Embarked.fillna('S',inplace=True)
test.Embarked.fillna('S',inplace=True)


# 2) age -> category age

# In[ ]:


def categorize_age(age):
    if age < 10:
        return 0
    elif age < 20:
        return 1
    elif age < 30:
        return 2
    elif age < 40:
        return 3
    elif age < 50:
        return 4
    elif age < 60:
        return 5
    else:
        return 6


# In[ ]:


train['Category_age'] = train['Age'].apply(categorize_age)
test['Category_age'] = test['Age'].apply(categorize_age)


# In[ ]:


bar_plot('Category_age')


# string data -> number 

# In[ ]:


test.Initial.unique()


# In[ ]:


train['Initial_no'] = train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rev':4, 'Dr':5, 'etc': 6})
test['Initial_no'] = test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rev':4, 'Dr':5,'etc': 6})


# In[ ]:


train['Embarked_no'] = train['Embarked'].map({'S':0,'C':1,'Q':1})
test['Embarked_no'] = test['Embarked'].map({'S':0,'C':1,'Q':1})


# In[ ]:


train['Sex_no'] = train['Sex'].map({'male':0, 'female':1})
test['Sex_no'] = test['Sex'].map({'male':0, 'female':1})


# In[ ]:


train.head()


# In[ ]:


feature = ['Pclass','SibSp','Parch','Category_age','Initial_no','Embarked_no','Sex_no']
X = train[feature]
test_X = test[feature]
y = train.Survived


# In[ ]:


X.head()


# ## Baseline Model
# train and check

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


print(train_X.shape)
print(valid_X.shape)


# In[ ]:


model = RandomForestClassifier(n_estimators=150, random_state=0)
model.fit(train_X, train_y)


# In[ ]:


predict = model.predict(valid_X)


# In[ ]:


accuracy = metrics.accuracy_score(valid_y, predict)
accuracy


# ## create submission file

# In[ ]:


model.fit(X,y)
predict = model.predict(test_X)
predict.shape


# In[ ]:


submission = pd.DataFrame([test.PassengerId, predict]).T
submission.columns = ['PassengerId','Survived']
submission.to_csv('base_submission.csv',index=False)


# In[ ]:


submission


# In[ ]:




