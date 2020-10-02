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


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# Let's take a look at how many rows are null in each column. 

# In[ ]:


train.isnull().sum()


# From above, we can see that age and cabin have several null rows. We will remove the cabin and ticket columns. We cannot replace the NaN columns in the Cabin column since the values are not numeric, and we do not know what cabin each person stayed in. We will remove the ticket column due to the fact every person has a unique ticket id. We will also remove the Name column, since it is not relevant.

# In[ ]:


train.drop(['Name','Ticket', 'Cabin'], axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.drop(['Name','Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# Now, we will fill the null columns for age. We can simply take the median age and use it to fill in the nulls. we will do the same for Fare. As for the null in Embark, we will use the label 'S'.

# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['Embarked'].fillna('S', inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)


# Now we can start to do some feature engineering. We know that there are several ages within the Age column. In order to remedy this, we should dived the ages into buckets. The youngest person in the dataset is younger than 1 year old. The oldest person is 76 years old. We can devide the ages into buckets containing 20 years. The buckets will be 20 and younger, between 20 and 40, and 60+.

# In[ ]:


train.loc[ train['Age'] <= 16, 'Age']                        = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age']                         = 4

test.loc[ test['Age'] <= 16, 'Age']                       = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age']                        = 4


# In[ ]:


train.head()


# In[ ]:


test.head()


# The same will be done for Fare.

# In[ ]:


train.loc[ train['Fare'] <= 7.91, 'Fare']                             = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare']                                = 3
train['Fare'] = train['Fare'].astype(int)

test.loc[ test['Fare'] <= 7.91, 'Fare']                            = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare']                               = 3
test['Fare'] = test['Fare'].astype(int)


# In[ ]:


train.loc[ train['Embarked'] == 'C', 'Embarked'] = 0
train.loc[ train['Embarked'] == 'Q', 'Embarked'] = 1
train.loc[ train['Embarked'] == 'S', 'Embarked'] = 3

test.loc[ test['Embarked'] == 'C', 'Embarked'] = 0
test.loc[ test['Embarked'] == 'Q', 'Embarked'] = 1
test.loc[ test['Embarked'] == 'S', 'Embarked'] = 3


# In[ ]:


train.loc[ train['Sex'] == 'Female', 'Sex'] = 0
train.loc[ train['Sex'] == 'Male', 'Sex'] = 1

test.loc[ test['Sex'] == 'Female', 'Sex'] = 0
test.loc[ test['Sex'] == 'Male', 'Sex'] = 1


# Now we can add the Parch and SibSp columns together to determine if a passenger was part of a family or not. The 

# In[ ]:


train['Family'] = train['Parch'] + train['SibSp']

test['Family'] = test['Parch'] + test['SibSp']


# In[ ]:


train = pd.concat([train, pd.get_dummies(train['Pclass'], prefix='Pclass'),
                     pd.get_dummies(train['Sex'], prefix='Sex'),
                     pd.get_dummies(train['Family'], prefix='Family'),
                     pd.get_dummies(train['Embarked'], prefix='Embarked'),
                     pd.get_dummies(train['Age'], prefix='Age'),
                     pd.get_dummies(train['Fare'], prefix='Fare')],
                    axis=1)
test = pd.concat([test, pd.get_dummies(test['Pclass'], prefix='Pclass'),
                     pd.get_dummies(test['Sex'], prefix='Sex'),
                     pd.get_dummies(test['Family'], prefix='Family'),
                     pd.get_dummies(test['Embarked'], prefix='Embarked'),
                     pd.get_dummies(test['Age'], prefix='Age'),
                     pd.get_dummies(test['Fare'], prefix='Fare')],
                    axis=1)


# In[ ]:


train.drop(['Pclass',  'Sex', 'Age', 'Fare', 'SibSp','Parch', 'Embarked', 'PassengerId',  'Family'], axis=1, inplace=True)
test.drop(['Pclass', 'Sex', 'Age', 'Fare', 'SibSp','Parch', 'Embarked', 'PassengerId',  'Family'], axis=1, inplace=True)


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head(25)


# In[ ]:


train.columns, test.columns


# In[ ]:


y = train['Survived']
train.drop('Survived', axis=1, inplace=True);


# In[ ]:


train.shape, test.shape


# We will now look at several different classification models to determine what model performs the best.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[ ]:


rf = RandomForestClassifier()
rf.fit(train, y)
y_rf = rf.predict(test)
rf.score(train, y) 


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train, y)


# In[ ]:


dt_y = dt.predict(test)
dt.score(train,y)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
lr.fit(train, y)


# In[ ]:


lr_y = lr.predict(test)
lr.score(train,y)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train, y)


# In[ ]:


knn_y = knn.predict(test)
knn.score(train,y)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(train,y)


# In[ ]:


nb_y = nb.predict(test)
nb.score(train,y)


# In[ ]:


from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(train, y)


# In[ ]:


svc_y = svc.predict(test)
svc.score(train,y)


# In[ ]:





# In[ ]:


final= pd.DataFrame()
test2 = pd.read_csv('/kaggle/input/titanic/test.csv')
final['PassengerId'] = test2['PassengerId']
final['Survived'] = dt_y
final.to_csv('submission.csv',index=False)


# In[ ]:




