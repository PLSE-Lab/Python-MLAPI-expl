#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
titanic = pd.read_csv("../input/train.csv") 
titanic_t = pd.read_csv("../input/test.csv")


# In[ ]:


titanic.head()


# In[ ]:


titanic['Sex'].value_counts()


# In[ ]:


titanic.info()


# In[ ]:


titanic.describe()


# # Data Preprocessing

# In[ ]:


titanic = titanic.drop(['Cabin'], axis=1)
titanic_t = titanic_t.drop(['Cabin'], axis=1)


# In[ ]:


plt.figure(figsize = (20,10))
plt.hist(titanic['Age'], bins = 50)


# In[ ]:


fare_mean = titanic['Fare'].mean()


# In[ ]:


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


# In[ ]:


titanic['Age'] = titanic[['Age','Pclass']].apply(impute_age,axis=1)
titanic_t['Age'] = titanic_t[['Age','Pclass']].apply(impute_age,axis=1)
# fill na in Age with meidan
titanic_t['Fare'] = titanic_t['Fare'].fillna(fare_mean)


# In[ ]:


titanic['Embarked'].value_counts()


# In[ ]:


#fill na in Embarked with most frequent value
titanic['Embarked'] = titanic['Embarked'].fillna('S')
#fill na in Embarked with most frequent value
titanic_t['Embarked'] = titanic_t['Embarked'].fillna('S')


# In[ ]:


# make sure that there are no null values
titanic.info()


# In[ ]:


titanic.head()


# ### Drop 'Ticket' and PassengerID' columns

# In[ ]:


titanic = titanic.drop(['PassengerId', 'Ticket'], axis=1)
titanic_t = titanic_t.drop(['PassengerId', 'Ticket'], axis=1)


# # Feature Engineering

# In[ ]:


titanic.head()


# In[ ]:


titanic['With_someone'] = titanic['SibSp'] | titanic['Parch']
titanic_t['With_someone'] = titanic_t['SibSp'] | titanic_t['Parch']


# In[ ]:


titanic['With_someone'] = titanic['With_someone'].apply(lambda x:1 if x >=1 else 0)
titanic_t['With_someone'] = titanic_t['With_someone'].apply(lambda x:1 if x >=1 else 0)


# In[ ]:


titanic['With_someone'].unique()


# In[ ]:


titanic['Title'] = titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
titanic_t['Title'] = titanic_t['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
titanic['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

titanic['Title'] = titanic['Title'].map(title_mapping)
titanic_t['Title'] = titanic_t['Title'].map(title_mapping)


# In[ ]:


titanic = titanic.drop(['Name'], axis=1)
titanic_t = titanic_t.drop(['Name'], axis=1)


# In[ ]:


plt.hist(titanic['SibSp'])


# In[ ]:


plt.hist(titanic['Parch'])


# In[ ]:


titanic['family members'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic_t['family members'] = titanic_t['SibSp'] + titanic_t['Parch'] + 1


# In[ ]:


sns.distplot(titanic['Age'])


# Age stages from: https://hhpolishinginc.com/74084_erik_erikson_developmental_stages_chart_ideas/

# In[ ]:


def age(x):
    if x < 5:
        return 'infant'
    elif x < 19:
        return 'adolescence'
    elif x < 35:
        return 'young adulthood'
    elif x < 51:
        return 'adulthood'
    else:
        return 'Mature adulthood'


# In[ ]:


titanic.head()


# In[ ]:


titanic.head()


# In[ ]:


plt.figure(figsize = (20, 5))
sns.distplot(titanic['Fare'])


# In[ ]:


titanic[titanic['Fare'] > 100].shape


# In[ ]:


def fare(x):
    if x == 0:
        return 'Stowaway'
    if x < 7:
        return 'cheap'
    elif x < 10:
        return 'mid-cheap'
    elif x < 15:
        return 'high cheap'
    elif x < 20:
        return 'medium'
    elif x < 50:
        return 'high'
    else:
        return 'very high'


# In[ ]:


titanic.head()


# In[ ]:


titanic.head()


# In[ ]:


titanic.head()


# # One hot Encoding

# In[ ]:


titanic = pd.get_dummies(titanic, columns = ['Pclass', 'Sex', 'Embarked', 'Title'], drop_first = True)
titanic_t = pd.get_dummies(titanic_t, columns = ['Pclass', 'Sex', 'Embarked', 'Title'], drop_first = True)


# In[ ]:


titanic.head()


# In[ ]:


titanic = titanic.drop(['SibSp', 'Parch', 'Age'], axis=1)
titanic_t = titanic_t.drop(['SibSp', 'Parch', 'Age'], axis=1)


# In[ ]:


X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']


# In[ ]:


X_t = titanic_t


# # Feature Scaling

# In[ ]:


X.head()


# In[ ]:


X_scale = X[['Fare', 'family members']]
X_noscale = X.drop(['Fare', 'family members'], axis=1)

X_scale_t = X_t[['Fare', 'family members']]
X_noscale_t = X_t.drop(['Fare', 'family members'], axis=1)


# In[ ]:


X_noscale.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_scaled = sc_X.fit_transform(X_scale)
X_scaled_t = sc_X.fit_transform(X_scale_t)


# In[ ]:


X_scaled = pd.DataFrame(X_scaled, columns=['Fare', 'family members'])
X_scaled_t = pd.DataFrame(X_scaled_t, columns=['Fare', 'family members'])


# In[ ]:


X = pd.concat([X_scaled, X_noscale], axis=1)
X_t = pd.concat([X_scaled_t, X_noscale_t], axis=1)


# In[ ]:


X.head()


# # Classification model

# ## KNN

# In[ ]:


k_range = [4]
weight_options = ['uniform']
norm = [1]
algo = ['ball_tree']


# In[ ]:


param_grid = dict(n_neighbors = k_range, weights = weight_options, p = norm, algorithm = algo)
param_grid


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[ ]:


grid_knn = GridSearchCV(knn, param_grid, cv = 10, scoring='accuracy', return_train_score=False)
grid_knn.fit(X, y)


# In[ ]:


grid_knn.best_params_


# In[ ]:


grid_knn.best_score_


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()


# In[ ]:


param_grid = dict(n_estimators = [10], criterion = ['gini'], max_depth = [135, 140, 145])


# In[ ]:


grid_forest = GridSearchCV(forest_clf, param_grid, cv = 10, scoring='accuracy', return_train_score=False)
grid_forest.fit(X, y)


# In[ ]:


grid_forest.best_params_


# In[ ]:


grid_forest.best_score_


# ## SVM

# In[ ]:


from sklearn import svm
clf = svm.SVC(probability = False)


# In[ ]:


param_grid = dict(C = [34], kernel = ['poly'], gamma = ['scale'], degree = [2])
param_grid


# In[ ]:


grid_svm = GridSearchCV(clf, param_grid, cv = 10, scoring='accuracy', return_train_score=True)
grid_svm.fit(X, y)


# In[ ]:


grid_svm.best_params_


# In[ ]:


grid_svm.best_score_


# ## logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


clf_lr = LogisticRegression()


# In[ ]:


param_grid = dict(penalty = ['l2'], solver = ['newton-cg', 'lbfgs'])
param_grid


# In[ ]:


grid_lr = RandomizedSearchCV(clf_lr, param_grid, cv = 10, scoring='accuracy', return_train_score=True, n_iter=150)
grid_lr.fit(X, y)


# In[ ]:


grid_lr.best_params_


# In[ ]:


grid_lr.best_score_


# ## Decision tree (try to overfit) 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf_dt = DecisionTreeClassifier()


# In[ ]:


param_grid = dict(criterion = ['entropy'], max_depth = [6])
param_grid
#[2, 5, 10, 30, 50, 100]


# In[ ]:


grid_tree = GridSearchCV(clf_dt, param_grid, cv = 10, scoring='accuracy', return_train_score=False)
grid_tree.fit(X, y)


# In[ ]:


grid_tree.best_params_


# In[ ]:


grid_tree.best_score_


# ## Gradient Boost

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


clf_gb = GradientBoostingClassifier()


# In[ ]:


param_grid = dict(loss = ['exponential'], learning_rate = [0.2], n_estimators = [21], max_depth = [4])
param_grid


# In[ ]:


grid_gb = GridSearchCV(clf_gb, param_grid, cv = 10, scoring='accuracy', return_train_score=False)
grid_gb.fit(X, y)


# In[ ]:


grid_gb.best_params_


# In[ ]:


grid_gb.best_score_


# ## Ada Boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


clf_ab = AdaBoostClassifier()


# In[ ]:


param_grid = dict(n_estimators = [65], algorithm = ['SAMME'])
param_grid


# In[ ]:


grid_ab = GridSearchCV(clf_ab, param_grid, cv = 10, scoring='accuracy', return_train_score=False)
grid_ab.fit(X, y)


# In[ ]:


grid_ab.best_params_


# In[ ]:


grid_ab.best_score_


# # Ensemble hard Voting

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


eclf_hard = VotingClassifier(estimators = [('knn', grid_knn), ('forest', grid_forest), ('svm', grid_svm),
                                     ('Logistic', grid_lr), ('tree', grid_tree),
                                      ('GradientBoost', grid_gb), 
                                          ('AdaBoost', grid_ab)], voting='hard', weights=[2, 1, 2.5, 0.8, 3, 3.5, 1])


# In[ ]:


eclf_hard = eclf_hard.fit(X, y)


# # Prediction and Submission

# In[ ]:


y_pred = eclf_hard.predict(X_t)


# In[ ]:


sub = pd.DataFrame(y_pred, columns=['Survived'])


# In[ ]:


titanic = pd.read_csv("../input/test.csv")


# In[ ]:


submission = pd.concat([titanic['PassengerId'], sub], axis=1)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('sub.csv', index=False)

