#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data_filename = '../input/titanic/train.csv'
test_data_filename = '../input/titanic/test.csv'

titanic_train_data = pd.read_csv(train_data_filename, index_col='PassengerId')
titanic_test_data = pd.read_csv(test_data_filename, index_col='PassengerId')
display(titanic_train_data.head())
display(titanic_test_data.head())


# In[ ]:


titanic_train_data.info()


# In[ ]:


titanic_train_data.Sex.value_counts()


# In[ ]:


labelencoder = LabelEncoder()

titanic_train_data['Sex'] = labelencoder.fit_transform(titanic_train_data.Sex)
titanic_test_data['Sex'] = labelencoder.fit_transform(titanic_test_data.Sex)
display(titanic_train_data.head())
display(titanic_test_data.head())


# In[ ]:


titanic_train_data['Title'] = titanic_train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
titanic_test_data['Title'] = titanic_test_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
display(titanic_train_data.Title)
display(titanic_test_data.Title)


# In[ ]:


titanic_test_data[titanic_test_data.Age.isna()]


# In[ ]:


titanic_train_data['Age'] = titanic_train_data.Age.interpolate(method='akima')
titanic_test_data['Age'] = titanic_test_data.Age.interpolate(method='akima')
titanic_test_data['Age'].fillna(30, inplace=True)
display(titanic_test_data.info())
display(titanic_train_data.info())


# In[ ]:


titanic_train_data['age_range'] = pd.cut(titanic_train_data.Age, bins=[0,2,17,65,99], labels=['Baby', 'Child', 'Adult', 'Elderly'])
titanic_test_data['age_range'] = pd.cut(titanic_test_data.Age, bins=[0,2,17,65,99], labels=['Baby', 'Child', 'Adult', 'Elderly'])
display(titanic_train_data.age_range)
display(titanic_test_data.age_range)


# In[ ]:


titanic_train_data.loc[titanic_train_data.age_range.isna(), 'Title'].unique()


# In[ ]:


titles = {'Mr': 'Adult', 'Mrs': 'Adult', 'Miss': 'Child', 'Master': 'Child', 'Dr': 'Adult', 'Ms': 'Adult'}
title_age_map = titanic_train_data.loc[titanic_train_data.age_range.isna(), 'Title'].map(titles)
titanic_train_data.loc[titanic_train_data.age_range.isna(), 'age_range'] = title_age_map


# In[ ]:


titanic_test_data.loc[titanic_test_data.age_range.isna(), 'Title'].unique()


# In[ ]:


title_age_map = titanic_test_data.loc[titanic_test_data.age_range.isna(), 'Title'].map(titles)
titanic_test_data.loc[titanic_test_data.age_range.isna(), 'age_range'] = title_age_map


# In[ ]:


titanic_train_data['Age'] = titanic_train_data.Age.astype('int')
titanic_train_data.head(20)


# In[ ]:


labelencoder = LabelEncoder()
titanic_train_data['Embarked'].fillna('X', inplace=True)
titanic_train_data['Embarked'] = labelencoder.fit_transform(titanic_train_data.Embarked)
titanic_test_data['Embarked'].fillna('X', inplace=True)
titanic_test_data['Embarked'] = labelencoder.transform(titanic_test_data.Embarked)
titanic_train_data.head()


# In[ ]:


le_age_range = LabelEncoder()
titanic_train_data['age_range'] = le_age_range.fit_transform(titanic_train_data.age_range)
#titanic_test_data['age_range'] = le_age_range.fit_transform(titanic_test_data.age_range)
titanic_train_data.head()


# In[ ]:


#titanic_train_data['age_range'] = le_age_range.fit_transform(titanic_train_data.age_range)
titanic_test_data['age_range'] = le_age_range.transform(titanic_test_data.age_range)
titanic_test_data.head()


# In[ ]:


titanic_test_data.age_range.unique()


# In[ ]:


display(titanic_train_data.corr())


# In[ ]:


sns.swarmplot(x=titanic_train_data.Survived, y=titanic_train_data.age_range)


# In[ ]:


sns.distplot(a=titanic_train_data.Age, bins=8)


# In[ ]:


titanic_train_data.head(20)


# In[ ]:


titanic_train_data.drop(columns=['Cabin', 'Ticket'], inplace=True)
titanic_train_data.head()


# In[ ]:


print(titanic_train_data.info())


# In[ ]:


y = titanic_train_data.Survived
features = ['Sex', 'Fare', 'Pclass', 'Age']
X = titanic_train_data.drop(['Survived', 'Name'] , axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

numeric_cols = ~X.dtypes.isin(["object", 'category'])
categorical_cols = ~numeric_cols
# chain preprocessing into a Pipeline object
# each step is a tuple of (name you chose, sklearn transformer)
numeric_preprocessing_steps = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='median')),
    ('standard_scaler', StandardScaler())
])

category_preprocessing_steps = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(drop='if_binary'))
])

# create the preprocessor stage of final pipeline
# each entry in the transformer list is a tuple of
# (name you choose, sklearn transformer, list of columns)
preprocessor = ColumnTransformer(
    transformers = [
        ("numeric", numeric_preprocessing_steps, numeric_cols),
        ("categorical", category_preprocessing_steps, categorical_cols)
    ],
    remainder = "drop"
)
print(X.dtypes.isin(["object", 'category']))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
steps = [('scaler', StandardScaler())]
# pipe = Pipeline(steps)
# pipe = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])
pipe = Pipeline([('pre', preprocessor), ('classifier', RandomForestClassifier())])
print(pipe.get_params())
# Create param grid.

param_grid = [
    {'classifier': [RandomForestClassifier(random_state=1)],
     'classifier__n_estimators' : [500],
     'classifier__max_features' : ['auto'],
     'classifier__min_samples_leaf': [1],
     'classifier__max_depth': [8],
     # 'classifier__oob_score': [True, False],
     'classifier__criterion': ['entropy']}
#     { 'classifier': [XGBClassifier()],
#       'classifier__xgbclassifier__n_estimators': [200],
#       'classifier__xgbclassifier__learning_rate': [0.4],
#       'classifier__xgbclassifier__subsample': [0.4, 0.6, 0.8],
#       'classifier__xgbclassifier__max_depth': [2, 3, 4],
#       'classifier__xgbclassifier__colsample_bytree': [0.8],
#       'classifier__xgbclassifier__min_child_weight': [1, 2, 3, 4]}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 10, verbose=True, n_jobs=-1)

# Fit on data

best_clf = clf.fit(x_train, y_train)


# In[ ]:


y_pred = best_clf.predict(x_test)
print(best_clf.score(x_test, y_test))
print(best_clf.best_estimator_)
print(best_clf.best_estimator_.named_steps["classifier"].feature_importances_)
print(best_clf.best_params_)
print(best_clf.best_score_)


# In[ ]:


# print(confusion_matrix(y_test, lr_pred))
# print(confusion_matrix(y_test, k_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


titanic_test_data.info()


# In[ ]:


titanic_test_data.drop(columns='Cabin', inplace=True)
titanic_test_data.Fare.fillna(0, inplace=True)


# In[ ]:


t = titanic_test_data[features]
titanic_test_data[features].isna()
Y_pred= best_clf.predict(t)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": titanic_test_data.index,
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




