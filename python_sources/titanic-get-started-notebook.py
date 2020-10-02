#!/usr/bin/env python
# coding: utf-8

# ## EDA & Machine Learning approach
# #### This is a get started notebook for kaggle titanic competition where all the best approaches has been used and this notebook is powered by DATACAMP.

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
        
for dirname, _, filenames in os.walk('/kaggle/output'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[ ]:


# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


df_train.head(2)


# In[ ]:


df_test.head(2)


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


sns.countplot(x='Survived', data=df_train);


# **Take-away: in the training set, less people survived than didn't. Let's then build a first model that predicts that nobody survived.**

# In[ ]:


df_test['Survived'] = 0


# In[ ]:


sns.countplot(x='Survived', data=df_test);


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('no_survivors.csv', index=False)


# In[ ]:


sns.countplot(x='Sex', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)


# **Take-away: Women were more likely to survive than men.**

# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


df_test['Survived'] = df_test.Sex == 'female'


# In[ ]:


df_test['Survived']


# In[ ]:


df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)


# In[ ]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# **Take-away: Passengers that travelled in first class were more likely to survive. On the other hand, passengers travelling in third class were more unlikely to survive.**

# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# **Take-away: Passengers that embarked in Southampton were less likely to survive.**

# In[ ]:


sns.distplot(df_train.Fare, kde=False);


# **Take-away: Most passengers paid less than 100 for travelling with the Titanic.**

# In[ ]:


df_train.groupby('Survived').Fare.hist(alpha=0.6);


# **Take-away: It looks as though those that paid more had a higher chance of surviving.**

# In[ ]:


df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=False);


# In[ ]:


sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True);


# In[ ]:


sns.swarmplot(x='Survived', y='Fare', data=df_train);


# **Take-away: Fare definitely seems to be correlated with survival aboard the Titanic.**

# In[ ]:


df_train.groupby('Survived').Fare.describe()


# In[ ]:


sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5});


# **Take-away: It looks like those who survived either paid quite a bit for their ticket or they were young.**

# In[ ]:


sns.pairplot(df_train_drop, hue='Survived')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Import data
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


survived_train = df_train.Survived


# In[ ]:


data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())


# In[ ]:


# Check out info of data
data.info()


# In[ ]:


data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[ ]:


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


data.info()


# In[ ]:


data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[ ]:


X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('1st_dec_tree.csv', index=False)


# * **https://www.datacamp.com/community/tutorials/kaggle-machine-learning-eda**
# * **https://www.datacamp.com/community/tutorials/kaggle-tutorial-machine-learning**
# * **https://www.datacamp.com/community/tutorials/machine-learning-python**
# * **https://www.datacamp.com/community/tutorials/machine-learning-python**
# * **https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python**
