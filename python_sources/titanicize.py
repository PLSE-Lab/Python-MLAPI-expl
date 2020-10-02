#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# In[ ]:


def get_last_name(df):
    df.loc[:, 'Name'] = df.Name.str.split(',', expand=True)[0]
    return df


# In[ ]:


def get_X_y(df, indicator):
    X = df.drop(INDICATOR, axis=1)
    y = df.loc[:, INDICATOR]
    return (X, y)


# In[ ]:


INDICATOR = 'Survived'


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.PassengerId.unique().shape


# In[ ]:


train.Survived.value_counts(dropna=False)


# In[ ]:


train.Pclass.value_counts(dropna=False)


# In[ ]:


train.Sex.value_counts(dropna=False)


# In[ ]:


train.Age.describe()


# In[ ]:


train.Age.value_counts(dropna=False)


# In[ ]:


train.SibSp.value_counts(dropna=False)


# In[ ]:


train.Parch.value_counts(dropna=False)


# In[ ]:


train.Fare.value_counts(dropna=False)


# In[ ]:


train.Fare.describe()


# In[ ]:


train.Cabin.value_counts(dropna=False)


# In[ ]:


train.Ticket.value_counts(dropna=False)


# In[ ]:


train.Name.isna().sum()


# In[ ]:


train.Embarked.value_counts(dropna=False)


# In[ ]:


X, y = get_X_y(train, INDICATOR)


# In[ ]:



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

name_transformer = Pipeline(steps=[
    ('get_last', FunctionTransformer(get_last_name, validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

binning_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('binner', KBinsDiscretizer(n_bins=8, encode='ordinal'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[ ]:


numeric_features = ['Fare']
binning_features = ['Age']
name_features = ['Name']
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']


# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('name', name_transformer, name_features),
        ('bin', binning_transformer, binning_features),
        ('cat', cat_transformer, categorical_features)]
)


# In[ ]:


rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random_forest', RandomForestClassifier(100))
])


cross_val_score(rf, X, y, cv=5).mean()


# In[ ]:


rf.fit(X, y)


# In[ ]:


y_pred = rf.predict(X)


# In[ ]:


precision_recall_fscore_support(y, y_pred, average='binary')


# In[ ]:


accuracy_score(y, y_pred)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()


# In[ ]:


(tn, fp, fn, tp)


# In[ ]:


train_copy = train.copy()


# In[ ]:


train_copy.loc[:, 'y_pred'] = y_pred


# In[ ]:


fps = train_copy[(train_copy.Survived == 0) & (train_copy.y_pred == 1)]
fns = train[(train_copy.Survived == 1) & (train_copy.y_pred == 0)]


# In[ ]:


fns


# In[ ]:


fps


# ## Model Selection

# In[ ]:


classifiers = [
    KNeighborsClassifier(),
    SVC(), 
    LinearSVC(), 
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(100), 
    AdaBoostClassifier(), 
    GradientBoostingClassifier(),
    LGBMClassifier(),
    XGBClassifier(),
    LogisticRegression(),
    SGDClassifier()
]


# In[ ]:


overall = []
for i in range(5):
    best = None
    for classifier in classifiers:
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        print('++++++++++++++++++++++++++++++++++++++++++++++++')
        print(classifier)
        print("Mean Accuracy for K=5: ")
        score = cross_val_score(rf, X, y, cv=5).mean()
        print(score)
        if not best or score > best[1]:
            best = (classifier, score)
    overall.append(best)


# In[ ]:


overall


# ## LightGBM HPO

# In[ ]:


from sklearn.model_selection import GridSearchCV
lgbmc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lgbmc', LGBMClassifier())
])

params = {
    'lgbmc__max_bin': [260, 300, 350, 400],
    'lgbmc__learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'lgbmc__num_iterations': [150, 200, 300, 400, 500],
    'lgbmc__boosting': ['gbdt', 'dart']
}

grid = GridSearchCV(lgbmc, cv=5, n_jobs=-1, param_grid=params, verbose=1)
                    
grid.fit(X, y)


# ## Create Submission

# In[ ]:


X, y = get_X_y(train, INDICATOR)


# In[ ]:


lgbmc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lgbmc', LGBMClassifier())
])

lgbmc.fit(X, y)


# In[ ]:


y_pred_test = lgbmc.predict(test)


# In[ ]:


submission = test.copy()


# In[ ]:


submission.loc[:, 'Survived'] = y_pred_test


# In[ ]:


submission = submission.loc[:, ['PassengerId', 'Survived']]


# In[ ]:


submission.to_csv('submission.csv', index=False)

