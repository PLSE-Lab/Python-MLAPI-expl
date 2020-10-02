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


train = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# **Let's deal with these features and figure out which one is needed in prediction of survival
#   There is no relation of survival with Name, PassengerID and Ticket these are some sort of unique Entry for every passenger so first we drop     these columns**

# In[ ]:


train.drop(['Name','PassengerId','Ticket'], axis =1, inplace = True)


# In[ ]:


train.isnull().sum()


# AS WE SEE THERE IS HIGH NUMBER OF NULL VALUES IN CABIN COLUMNS SO WE WILL DROP THIS COLUMN INSTEAD OF USING DROPNA BECUASE BY DROPNA WE LOOSE LOTS OF DATA

# In[ ]:


train.drop('Cabin', axis =1, inplace = True)


# In[ ]:


train.corr()


# In[ ]:


import seaborn as sns
sns.heatmap(train.corr())


# AS WE HAVE SOME MISSING DATA IN AGE SO LETS FILL THESE WITH MEAN OF AGE COLUMNS

# In[ ]:


train['Age'] = train['Age'].fillna(np.mean(train['Age']))


# In[ ]:


train.isnull().sum()


# we have only 2 mising in Embarked so lets drop these rows
train.dropna(inplace = True)
# In[ ]:


train


# **Time to visualize **

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(train['Survived'])


# **From this count plot we see that the number to person died is more than survived**

# In[ ]:


plt.figure(figsize = (20,8))
sns.heatmap(train.corr(), annot = True)


# **Correlation is not much high in between**

# In[ ]:


plt.figure(figsize = (15,8))
sns.scatterplot(train['Survived'],train['Fare'])


# In[ ]:


print(train['Pclass'].unique())
print(train['Embarked'].unique())


# Lets go for categorical columns

# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
p = pd.get_dummies(train['Pclass'], drop_first = True)


# In[ ]:


train = pd.concat([sex,train,embark,p], axis =1)


# In[ ]:


train.drop(['Pclass','Sex','Embarked'], axis =1, inplace = True)


# Now we all set for training our model

# I am using Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


x = train.drop('Survived', axis =1)
y = train['Survived']


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(x,y)


# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
test_x = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test['Age'] = test['Age'].fillna(np.mean(test['Age']))


# In[ ]:


test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))


# In[ ]:


test.isnull().sum()


# In[ ]:


sex_t = pd.get_dummies(test['Sex'], drop_first = True)
embark_t = pd.get_dummies(test['Embarked'], drop_first = True)
p_t = pd.get_dummies(test['Pclass'], drop_first = True)


# In[ ]:


test = pd.concat([sex_t,test,embark_t,p_t], axis =1)


# In[ ]:


test.drop(['PassengerId','Name','Sex','Pclass','Ticket','Embarked','Cabin'], axis =1, inplace = True)


# In[ ]:


test


# In[ ]:


prediction = model.predict(test)


# In[ ]:


output = pd.DataFrame({'PassengerId' : test_x.PassengerId, 'Survived':prediction})
output.to_csv('Submission.csv', index = False)
output.head()


# for Logistic Regression i got 16k rank so used Random Forest Classifier

# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


n_estimators = [int(x) for x in np.linspace(20,400,num = 20)]
max_depth = [int(x) for x in np.linspace(1,100,num = 10)]
min_samples_split = [2,3,5,7,8]
min_samples_leaf = [2,3,5,8]
bootstrap = [True,False]


# In[ ]:


random_para = {'n_estimators' : n_estimators,
               'max_depth' : max_depth,
                'min_samples_split' : min_samples_split,
                'min_samples_leaf' : min_samples_leaf,
                'bootstrap' : bootstrap}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


random = RandomizedSearchCV(estimator = rf, param_distributions = random_para, cv = 3, n_iter = 100, verbose = 5)


# In[ ]:


random.fit(x,y)


# In[ ]:


random.best_estimator_


# In[ ]:


rf = RandomForestClassifier(max_depth=34, min_samples_leaf=3, min_samples_split=3,
                       n_estimators=200)


# In[ ]:


rf.fit(x,y)


# In[ ]:


prediction1 = rf.predict(test)


# In[ ]:


output = pd.DataFrame({'PassengerId' : test_x.PassengerId, 'Survived':prediction1})
output.to_csv('Submission.csv', index = False)
output.head()


# In[ ]:




