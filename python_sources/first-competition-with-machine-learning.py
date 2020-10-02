#!/usr/bin/env python
# coding: utf-8

# This is my first time to challenge a competition without helping.
# Also, since I want to become a data analyst or data scientist in future, I want to practice the skills of explaination and maybe to build a blog.

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


from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(strategy = 'median')
mostfreq_imputer = SimpleImputer(strategy = 'most_frequent')


# First, we need to read and understand our data.

# In[ ]:


test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
combine = [test_data, train_data]
train_data.head()


# In[ ]:


train_data.tail()


# * sibsp means the number of siblings or spouses aboard the ship.
# * parch means the number of parents or children aboard the ship.
# * embarked means the port of embarkation.
# 
# In here, we can see that there are some missing data.
# Also, we can see that sex and embarked are categories.
# 
# Before we preprocess these data, we need more information about it.

# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# Let's check those missing data

# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.describe(include=['O'])


# Analysis the data First

# In[ ]:


cols = train_data.columns
dis_cols = cols.drop(['Survived', 'PassengerId','Name','Age','Ticket','Fare','Cabin'])
for x in dis_cols:
    ana = train_data[[x, 'Survived']].groupby(x).mean().sort_values(by='Survived',ascending=False)
    print(ana)
    print("_"*10)


# From here, we can see that if we guess that all women were alive and all men were dead, there are 74% chances win.
# Therefore, we need to create a model that winning chance better than 74% 
# 
# Also, we can see that survive rate seems like related to Pclass.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
g = sns.FacetGrid(train_data, col = "Survived", size = 10)
g.map(plt.hist, 'Age', bins = 50)
plt.xticks(np.arange(0,81, step = 5))
plt.show()
plt.figure(figsize = (14,6))
g = sns.FacetGrid(train_data, col = "Survived", size = 10)
g.map(plt.hist, 'Fare', bins = 20)
plt.xticks(np.arange(0,520, step = 50))
plt.show()


# I guess Pclass may be related to Fare. I need to prove my guess.

# In[ ]:


train_data.plot.scatter(x='Pclass', y='Fare', c='Survived', colormap='viridis')


# It seems that I was wrong about the relation between Fare and Pclass. However, we can see that the relation between Pclass and Survived are better than between Fare and Survived.

# we find out that when age is smaller than 12, the survived rate become bigger. On the other hand, when age is bigger than 40, the survived rate become smaller.

# Next, we start to clean our data. First we need to delete "cabin", since there are only 204 non-missing data out of 891. Also "ticket", since I cannot find any information in it.

# In[ ]:


train_data = train_data.drop(['Cabin','Ticket'], axis = 1)
test_data = test_data.drop(['Cabin', 'Ticket'], axis = 1)
combine = [train_data, test_data]
train_data.head()


# Next, I want to fill in those missing data.
# * Those in "Embarked" columns, I will fill in with most frequency
# * Those in "Fare", I will fill in with median.

# In[ ]:


train_data['Age'] = median_imputer.fit_transform(train_data[['Age']])
test_data['Age'] = median_imputer.transform(test_data[['Age']])
train_data['Fare'] = median_imputer.fit_transform(train_data[['Fare']])
test_data['Fare'] = median_imputer.transform(test_data[['Fare']])
train_data['Embarked'] = mostfreq_imputer.fit_transform(train_data[['Embarked']])
test_data['Embarked'] = mostfreq_imputer.transform(test_data[['Embarked']])


# Now, we need to deal with "SibSp" and "Parch". Actually, these two columns can be combined together as "family".

# In[ ]:


for df in combine:
    df['family'] = df['SibSp']+df['Parch']
train_data = train_data.drop(['SibSp','Parch'], axis = 1)
test_data = test_data.drop(['SibSp','Parch'],axis=1)
combine = [train_data, test_data]


# We can research whether or not there is some kind of relation between survive rate and number of family members aboard. 

# In[ ]:


train_data[['family', 'Survived']].groupby('family').mean().sort_values(by='Survived',ascending=False)


# It seems that the number of family aboard is not matter, but whether there is a family member aboard or not is matter.

# In[ ]:


for df in combine:
    df['Alone'] = 0
    df.loc[df['family']==0, 'Alone']=1
train_data = train_data.drop(['family'],axis = 1)
test_data = test_data.drop(['family'],axis = 1)
combine = [train_data, test_data]
train_data[['Alone', 'Survived']].groupby('Alone').mean().sort_values(by='Survived',ascending=False)


# Let's deal with "Name" and "Sex".

# In[ ]:


for df in combine:
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.',expand = False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# In[ ]:


for df in combine:
    df['Title'] = df['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Mlle','Mme','Ms','Rev','Sir'],'Others')
train_data[['Title','Survived']].groupby(['Title']).mean()


# In[ ]:


train_data = train_data.drop(['Name'],axis=1)
test_data = test_data.drop(['Name'],axis = 1)
combine = [train_data,test_data]
for df in combine:
    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
features = ['Embarked', 'Title']
OH_train_cols = pd.DataFrame(encoder.fit_transform(train_data[features]))
OH_test_cols = pd.DataFrame(encoder.transform(test_data[features]))
OH_train_cols.index = train_data.index
OH_test_cols.index = test_data.index
train_data = train_data.drop(features, axis = 1)
test_data = test_data.drop(features, axis = 1)
train_data = pd.concat([train_data, OH_train_cols], axis = 1)
test_data = pd.concat([test_data, OH_test_cols], axis = 1)


# Finally, let's train our model. In here, we use random forest.

# In[ ]:


from sklearn.model_selection import GridSearchCV
X = train_data.drop(['PassengerId','Survived'], axis = 1)
y = train_data['Survived']
X_test = test_data.drop(['PassengerId'], axis =1).copy()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
grid_feature = {'max_features':[10,11,12,13]}
model = RandomForestClassifier(n_estimators = 1000,
                             random_state=1,
                             n_jobs=-1)
grid = GridSearchCV(model, param_grid = grid_feature)
grid.fit(X,y)
print(grid.best_params_)
print(grid.best_score_)


# In[ ]:


model = RandomForestClassifier(n_estimators = 1000,
                               max_features = 12,
                             random_state=1,
                             n_jobs=-1)
model.fit(X, y)
y_test = model.predict(X_test)
model.score(X, y)


# In[ ]:


submission = pd.DataFrame({"PassengerId":test_data['PassengerId'],"Survived":y_test})
submission.to_csv('titanic_submission.csv', index = False)

