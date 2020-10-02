#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
sns.color_palette()

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_test = pd.read_csv("../input/test.csv")


# In[ ]:


df_test.head()


# The train and test set - Get the shape, columns, description of the columns.

# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# There are 891 records in train set and 418 in test set. Age, Cabin has null values in train set. Age, Cabin and Fare(1 record) has null values in test set.

# Analyse Survived

# In[ ]:


df_train['Survived'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x='Survived', data=df_train)
ax.set_title('Survived Distribution')
ax.set_xlabel('Survived')
ax.set_ylabel('Count')
plt.show()


# Male and Female distribution

# In[ ]:


df_train['Sex'].value_counts()


# In[ ]:


df_test['Sex'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Sex', data=df_train)
ax.set_title('Distribution by Sex')
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
plt.show()


# In[ ]:


df_train[df_train['Survived'] == 1]['Sex'].value_counts()


# In[ ]:


df_train[df_train['Survived'] == 0]['Sex'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Sex', data=df_train, hue='Survived')
ax.set_title('Survival by sex of traveller', fontsize=15)
ax.set_xlabel('Sex', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})


# In[ ]:


df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})


# Distribution by Class

# In[ ]:


df_train['Pclass'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Pclass', data=df_train)
ax.set_title('Passenger Distribution by class', fontsize=15)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


df_train[df_train['Survived'] == 1]['Pclass'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x="Pclass", data=df_train, hue='Survived')
ax.set_title('Survival by class')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
plt.show()


# Let's extract salutation from Name

# In[ ]:


df_train['Salutation'] = df_train['Name'].transform(lambda x : x.split(',')[1].split('.')[0])
df_train['Salutation'] = df_train['Salutation'].transform(lambda x: x.str.strip())


# In[ ]:


df_train['Salutation'].value_counts()


# In[ ]:


df_test['Salutation'] = df_test['Name'].transform(lambda x : x.split(',')[1].split('.')[0])
df_test['Salutation'] = df_test['Salutation'].transform(lambda x: x.str.strip())


# In[ ]:


df_test['Salutation'].value_counts()


# We can group the above salutations to **Mr**, **Miss**(Miss, Ms, Mlle), **Mrs**(Mrs, Mme), **Master,** **Other**(Dr, Rev, Col, Major, Sir, Don, Jonkheer, the Countess, Capt, Lady, Dona)

# In[ ]:


df_train['Salutation'] = df_train['Salutation'].replace(['Ms', 'Mlle'], 'Miss')


# In[ ]:


df_test['Salutation'] = df_test['Salutation'].replace(['Ms', 'Mlle'], 'Miss')


# In[ ]:


df_train['Salutation'] = df_train['Salutation'].replace('Mme', 'Mrs')


# In[ ]:


df_train['Salutation'] = df_train['Salutation'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Don', 
                        'Jonkheer', 'the Countess', 'Capt', 'Lady', 'Dona'], 'Other')


# In[ ]:


df_test['Salutation'] = df_test['Salutation'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Don', 
                        'Jonkheer', 'the Countess', 'Capt', 'Lady', 'Dona'], 'Other')


# In[ ]:


df_train['Salutation'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Salutation', data=df_train)
ax.set_title('Salutation/Title Distribution', fontsize=15)
ax.set_xlabel('Salutation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Salutation', data=df_train, hue='Survived')
ax.set_title('Salutation/Title Distribution', fontsize=15)
ax.set_xlabel('Salutation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


df_test['Salutation'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Salutation', data=df_test)
ax.set_title('Salutation/Title Distribution', fontsize=15)
ax.set_xlabel('Salutation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


df_train['Salutation'] = df_train['Salutation'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4})


# In[ ]:


df_test['Salutation'] = df_test['Salutation'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4})


# Identify the lone travellers. Travellers with 0 Parch and 0 SibSp are lone travellers.

# In[ ]:


df_train['Is_Alone'] = 0
for index, item in df_train.iterrows():
    if item['Parch'] ==0 and item['SibSp'] == 0:
        df_train.loc[index, 'Is_Alone'] = 1


# In[ ]:


df_train['Is_Alone'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.countplot(x='Is_Alone', data=df_train)
ax.set_title('Lone Travellers Distribution', fontsize=15)
ax.set_xlabel('Is_Alone', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# Check survived travellers whi travelled alone

# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x='Is_Alone', data=df_train, hue='Survived')
ax.set_title('Survival of Passengers travelled alone', fontsize=15)
ax.set_xlabel('Is_Alone', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


df_test['Is_Alone'] = 0
for index, item in df_test.iterrows():
    if item['Parch'] ==0 and item['SibSp'] == 0:
        df_test.loc[index, 'Is_Alone'] = 1


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.countplot(x='Is_Alone', data=df_test)
ax.set_title('Lone Travellers Distribution - Test', fontsize=15)
ax.set_xlabel('Is_Alone', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# Let's deal with travellers who had cabin. Let's change the cabin feature to 1 for travellers with cabin and zero for others.

# In[ ]:


df_train['Cabin'] = df_train['Cabin'].transform(lambda x: 0 if pd.isnull(x) else 1)


# In[ ]:


df_train['Cabin'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Cabin', data=df_train)
ax.set_title('Cabin Distribution', fontsize=15)
ax.set_xlabel('Has Cabin', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# Survival of travellers who had cabin

# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x='Cabin', data=df_train, hue='Survived')
ax.set_title('Survival of travellers with Cabin', fontsize=15)
ax.set_xlabel('Cabin', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


df_test['Cabin'] = df_test['Cabin'].transform(lambda x: 0 if pd.isnull(x) else 1)
df_test['Cabin'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Cabin', data=df_test)
ax.set_title('Cabin Distribution', fontsize=15)
ax.set_xlabel('Has Cabin', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# Let's handle embarked. Fill 2 null values with mode ('S')

# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna('S')


# In[ ]:


df_train['Embarked'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Embarked', data=df_train)
ax.set_title('Embarked Distribution')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Embarked', data=df_train, hue='Survived')
ax.set_title('Embarked Distribution')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()


# In[ ]:


df_test['Embarked'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Embarked', data=df_test)
ax.set_title('Embarked Distribution')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'S':1, 'Q':2})
df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'S':1, 'Q':2})


# Let's deal with Ticket feature. Ticket feature does not have any important information. So let's drop this feature.

# In[ ]:


df_train.drop('Ticket', inplace=True, axis=1)


# In[ ]:


df_test.drop('Ticket', inplace=True, axis=1)


# Let's check fare feature. Since the std deviation is high and the feature is continuous, let's fill the null value with median.

# In[ ]:


df_test['Fare'].describe()


# In[ ]:


df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())


# Let's map Fare to 4 quartiles using qcut.

# In[ ]:


pd.qcut(df_train['Fare'], 4).unique()


# In[ ]:


df_train.loc[df_train['Fare'] <= 7.91, 'Fare'] = 0
df_train.loc[((df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.454)), 'Fare'] = 1
df_train.loc[((df_train['Fare'] > 14.454) & (df_train['Fare'] <= 31)), 'Fare'] = 2
df_train.loc[df_train['Fare'] > 31, 'Fare'] = 3


# In[ ]:


df_test.loc[df_test['Fare'] <= 7.91, 'Fare'] = 0
df_test.loc[((df_test['Fare'] > 7.91) & (df_test['Fare'] <= 14.454)), 'Fare'] = 1
df_test.loc[((df_test['Fare'] > 14.454) & (df_test['Fare'] <= 31)), 'Fare'] = 2
df_test.loc[df_test['Fare'] > 31, 'Fare'] = 3


# In[ ]:


df_train['Fare'] = df_train['Fare'].astype('int')
df_test['Fare'] = df_test['Fare'].astype('int')
df_train['Fare'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Fare', data=df_train)
ax.set_title('Fare Distribution', fontsize=15)
ax.set_xlabel('Fare', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Fare', data=df_train, hue='Survived')
ax.set_title('Fare Distribution', fontsize=15)
ax.set_xlabel('Fare', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# The more the fare more people survived. Now finally let's deal with Age.

# In[ ]:


df_train['Age'].isna().sum()


# In[ ]:


df_test['Age'].isna().sum()


# Age feature has too many null values. Let's fill the values by taking the mean and stddev. [Inspiration taken from Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier)

# In[ ]:


age_avg   = df_train['Age'].mean()
age_std  = df_train['Age'].std()
age_null_count = df_train['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df_train['Age'][np.isnan(df_train['Age'])] = age_null_random_list
df_train['Age'] = df_train['Age'].astype(int)
    
pd.cut(df_train['Age'], 5).unique()


# In[ ]:


df_train.loc[df_train['Age']<=16, 'Age'] = 0
df_train.loc[((df_train['Age']>16)&(df_train['Age']<=32)), 'Age'] = 1
df_train.loc[((df_train['Age']>32)&(df_train['Age']<=48)), 'Age'] = 2
df_train.loc[((df_train['Age']>48)&(df_train['Age']<=64)), 'Age'] = 3
df_train.loc[df_train['Age']>64, 'Age'] = 4


# In[ ]:


age_avg   = df_test['Age'].mean()
age_std  = df_test['Age'].std()
age_null_count = df_test['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df_test['Age'][np.isnan(df_test['Age'])] = age_null_random_list
df_test['Age'] = df_test['Age'].astype(int)


# In[ ]:


df_test.loc[df_test['Age']<=16, 'Age'] = 0
df_test.loc[((df_test['Age']>16)&(df_test['Age']<=32)), 'Age'] = 1
df_test.loc[((df_test['Age']>32)&(df_test['Age']<=48)), 'Age'] = 2
df_test.loc[((df_test['Age']>48)&(df_test['Age']<=64)), 'Age'] = 3
df_test.loc[df_test['Age']>64, 'Age'] = 4


# In[ ]:


df_test.head()


# In[ ]:


df_train.drop('Name', axis=1, inplace=True)


# In[ ]:


df_test.drop('Name', axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.dtypes


# In[ ]:


df_test.dtypes


# # Let's build some classifiers

# Let's store the PassengerId for submitting predictions. Survived column for training 

# In[ ]:


PassengerId = df_test['PassengerId'].ravel()
y_all = df_train['Survived'].ravel()


# In[ ]:


df_train.drop('PassengerId', axis=1, inplace=True)
df_test.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


X_all = df_train.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(X_all.corr(), annot=True)
ax.set_title('Correlation of training set')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df_test.corr(), annot=True)
ax.set_title('Correlation of test set')
plt.show()


# # Model Creation

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# **Grid Search:**
# 1. Random Forest 
# 2. Extra Trees 
# 3. AdaBoost 
# 4. Gradient Boosting
# 5. SVM

# In[ ]:


acc_scorer = make_scorer(accuracy_score)
'''
clf = RandomForestClassifier()
rf_params = {
    "n_estimators": [100, 300, 500, 1000],
    "bootstrap": [True, False],
    "criterion": ['gini', 'entropy'],
    "warm_start": [True, False],
    "max_depth": [2, 4, 6],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [2, 4, 6]
}

clf = ExtraTreesClassifier()
xt_params = {
    "n_estimators":[100, 300, 500, 1000],
    "bootstrap": [True, False],
    "criterion": ['gini', 'entropy'],
    "warm_start": [True, False],
    "max_depth": [2, 4, 6],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [2, 4, 6]
}

clf = AdaBoostClassifier()
ad_params = {
    "n_estimators":[100, 300, 500, 1000],
    "learning_rate": [0.1, 0.3, 0.5, 0.75, 1]
}
clf = GradientBoostingClassifier()
gb_params = {
    "n_estimators":[100, 300, 500, 1000],
    "learning_rate": [0.1, 0.3, 0.5, 0.75, 1],
    "warm_start": [True, False],
    "max_depth": [2, 4, 6],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [2, 4, 6]
}
clf = SVC()
sv_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.01, 0.1, 1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]'''


# In[ ]:


#grid_search = GridSearchCV(clf, param_grid=sv_params, scoring=acc_scorer)
#grid_search.fit(X_train, y_train)


# In[ ]:


#grid_search.best_estimator_


# In[ ]:


rf_clf = RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=6, max_features='log2',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=4,
            min_samples_split=4, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=None, oob_score=False,
            random_state=42, verbose=0, warm_start=True)

et_clf = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
           max_depth=6, max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=42, verbose=0, warm_start=False)

ad_clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=300, random_state=42)

gb_clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=2,
              max_features='log2', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=2, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=300,
              n_iter_no_change=None, presort='auto', random_state=42,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)

sv_clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)


# In[ ]:


rf_clf.fit(X_train, y_train)
et_clf.fit(X_train, y_train)
ad_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)
sv_clf.fit(X_train, y_train)


# In[ ]:


rf_rank = rf_clf.feature_importances_
et_rank = et_clf.feature_importances_
ad_rank = ad_clf.feature_importances_
gb_rank = gb_clf.feature_importances_


# In[ ]:


df_feature_importance = pd.DataFrame({
    'Features': X_all.columns,
    'Random_Forest': rf_rank,
    'Extra_Trees': et_rank,
    'AdaBoost': ad_rank,
    'Gradient_Boost': gb_rank
})


# In[ ]:


df_feature_importance


# In[ ]:


fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['Random_Forest'])
ax.set_title('Random Forest Feature Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()


# In[ ]:


fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['Extra_Trees'])
ax.set_title('Extra Trees Feature Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()


# In[ ]:


fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['AdaBoost'])
ax.set_title('AdaBoost Feature Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()


# In[ ]:


fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['Gradient_Boost'])
ax.set_title('Gradient Boosting Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()


# In[ ]:


rf_pred = rf_clf.predict(X_test)
et_pred = et_clf.predict(X_test)
ad_pred = ad_clf.predict(X_test)
gb_pred = gb_clf.predict(X_test)
sv_pred = sv_clf.predict(X_test)


# In[ ]:


print('Random Forest Accuracy: {0:.2f}'.format(accuracy_score(y_test, rf_pred) * 100))
print('Extra Trees Accuracy: {0:.2f}'.format(accuracy_score(y_test, et_pred) * 100))
print('AdaBoost Accuracy: {0:.2f}'.format(accuracy_score(y_test, ad_pred) * 100))
print('Gradient Boosting Accuracy: {0:.2f}'.format(accuracy_score(y_test, gb_pred) * 100))
print('SVM Accuracy: {0:.2f}'.format(accuracy_score(y_test, sv_pred) * 100))


# Check the mean value of predictions of a classifier using KFold

# In[ ]:


def KFold_pred(clf, X_all, y_all):
    outcomes = []
    test_scores = []
    kf = KFold(n_splits=5, random_state=42, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(X_all)):
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        outcomes.append(acc)
        print('Fold {0} accuracy {1:.2f}'.format(i, acc))
    mean_accuracy = np.mean(outcomes)
    return mean_accuracy


# In[ ]:


rf_pred = KFold_pred(rf_clf, X_all, y_all)
print('Random Forest 5 folds mean accuracy: {0:.2f}'.format(rf_pred))
et_pred = KFold_pred(et_clf, X_all, y_all)
print('Extra Trees 5 folds mean accuracy: {0:.2f}'.format(et_pred))
ad_pred = KFold_pred(ad_clf, X_all, y_all)
print('AdaBoost 5 folds mean accuracy: {0:.2f}'.format(ad_pred))
gb_pred = KFold_pred(gb_clf, X_all, y_all)
print('Gradient Boosting 5 folds mean accuracy: {0:.2f}'.format(gb_pred))
sv_pred = KFold_pred(sv_clf, X_all, y_all)
print('SVM 5 folds mean accuracy: {0:.2f}'.format(sv_pred))


# **Out of fold average** : Out of fold average for the training and test set which will be used later for stacking

# In[ ]:


def oof_pred(clf, X_all, y_all, df_test):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    oof_train = np.zeros(X_all.shape[0])
    oof_test = np.zeros(df_test.shape[0])
    oof_test_kf = np.empty((kf.get_n_splits(), df_test.shape[0]))
    for i, (train_index, test_index) in enumerate(kf.split(X_all)):
        X_train = X_all.values[train_index]
        y_train = y_all[train_index]
        X_test = X_all.values[test_index]
        y_test = y_all[test_index]
        
        clf.fit(X_train, y_train)
        oof_train[test_index] = clf.predict(X_test)
        oof_test_kf[i, :] = clf.predict(df_test)
    oof_test = oof_test_kf.mean(axis=0)
    return oof_train, oof_test


# In[ ]:


rf_oof_train, rf_oof_test = oof_pred(rf_clf, X_all, y_all, df_test)
et_oof_train, et_oof_test = oof_pred(et_clf, X_all, y_all, df_test)
ad_oof_train, ad_oof_test = oof_pred(ad_clf, X_all, y_all, df_test)
gb_oof_train, gb_oof_test = oof_pred(gb_clf, X_all, y_all, df_test)
sv_oof_train, sv_oof_test = oof_pred(sv_clf, X_all, y_all, df_test)


# With Random Forest OOF values I got a public score of 0.80861. Let's use these outputs as new features and classify using LightGBM and XGBoost to check the improvement. 

# In[ ]:


base_level_train = pd.DataFrame({
    'Random_Forest':rf_oof_train,
    'Extra_Trees': et_oof_train,
    'AdaBoost': ad_oof_train,
    'Gradient_Boost':gb_oof_train,
    'Support_Vector':sv_oof_train
})


# In[ ]:


base_level_train.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(base_level_train.corr(), annot=True)
plt.show()


# In[ ]:


base_level_test = pd.DataFrame({
    'Random_Forest':rf_oof_test,
    'Extra_Trees': et_oof_test,
    'AdaBoost': ad_oof_test,
    'Gradient_Boost':gb_oof_test,
    'Support_Vector':sv_oof_test
})


# In[ ]:


base_level_test.head()


# In[ ]:


import xgboost as xgb
xg_clf = xgb.XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)


# In[ ]:


xg_clf.fit(base_level_train, y_all)
predictions = xg_clf.predict(base_level_test)


# In[ ]:


output = pd.DataFrame({ 'PassengerId' : PassengerId, 'Survived': predictions.astype(int) })
output.to_csv('titanic-predictions.csv', index = False)
output.tail()


# In[ ]:




