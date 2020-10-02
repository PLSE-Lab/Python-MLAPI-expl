#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
titanic_train_df = pd.read_csv('../input/train.csv')
titanic_test_df = pd.read_csv('../input/test.csv')
print(titanic_train_df.sample(2))
titanic_test_df.sample(2)


# In[ ]:


print(titanic_train_df.info())
titanic_test_df.info()


# In[ ]:


titanic_train_df['Family_Size']=titanic_train_df['SibSp']+titanic_train_df['Parch']
titanic_test_df['Family_Size']=titanic_test_df['SibSp']+titanic_test_df['Parch']
titanic_train_df.sample(5)


# In[ ]:


# dropping Name, ticket, cabin from both data set as this column is not required for analysis and could see the null values in Age, Fare, Cabin and Embarked columns,
# which will be replaced with mean or 3rd class info

titanic_train_df.drop(['Name','Ticket', 'Cabin'], axis=1, inplace=True)
titanic_test_df.drop(['Name','Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


# Filling NA value to Age with mean of the Age.
titanic_train_df.Age.fillna(titanic_train_df.Age.mean(), inplace=True)
titanic_test_df.Age.fillna(titanic_test_df.Age.mean(), inplace=True)


# In[ ]:


titanic_test_df.Fare.fillna(titanic_test_df[titanic_test_df.Pclass==3].Fare.mean(), inplace=True)


# In[ ]:


# Now the data set is ready by replacing the missing values. Now we need to replace Sex,Age,Embarked to categorical
titanic_train_df.info()


# In[ ]:


titanic_train_df['Sex'] = pd.Categorical(titanic_train_df['Sex']).codes
titanic_train_df['Embarked'] = pd.Categorical(titanic_train_df['Embarked']).codes
titanic_train_df.info()


# In[ ]:


titanic_df = pd.concat([titanic_train_df, titanic_test_df])

import matplotlib.pyplot as plt
names = titanic_train_df['Survived'].unique()
values = titanic_train_df['Survived'].value_counts().values

fig, ax = plt.subplots()
plt.xticks([0, 1])
plt.title('Survived ratio in training set')
plt.xlabel('0 - Dead, 1 - Survived')
plt.ylabel('Count rate')
for i, v in enumerate(values):
    ax.text(i, v+5, str(v), color='g', fontweight='bold')
ax.bar(names, values, color=(0.5,0.1,0.5,0.6),width=0.5)


# In[ ]:


titanic_train_df.info()


# In[ ]:


# The correlation shows that there is no relation between the independent and target variable ie. Survived column
# class and fare is negative marginally related.
titanic_df.corr()


# In[ ]:


plt.figure(figsize=(15,15))
import seaborn as sns
sns.heatmap(titanic_df.corr(),annot=True, vmin=-1, vmax=1, cmap='RdBu')


# In[ ]:


sns.pairplot(titanic_df)


# In[ ]:


titanic_test_df['Sex'] = pd.Categorical(titanic_test_df['Sex']).codes
titanic_test_df['Embarked'] = pd.Categorical(titanic_test_df['Embarked']).codes
titanic_test_df.info()


# In[ ]:


#Getting ready the train data
x_test_org = titanic_test_df
X_train = titanic_train_df.drop('Survived', axis=1)
y_train = titanic_train_df[['Survived']]
y_train = np.ravel(y_train)
print(X.info(), y.shape)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgbm
import catboost as cb


# Decision tree

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "criterion": ["gini", "entropy"],\n    "max_depth": [1, 2, 3, 5, 10, None], \n    "min_samples_split": [2, 3, 5, 10],\n    "min_samples_leaf": [1, 5, 10, 20]\n}\n\ntree_model = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, tree_model.predict(X_train)))\nprint(tree_model.best_score_)\nprint(tree_model.best_params_)\nprint(tree_model.best_estimator_)')


# Random forest

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "n_estimators": [2, 4, 5, 8, 10, 15], \n    "criterion": ["gini", "entropy"],\n    "max_features": ["auto", "log2"], \n    "max_depth": [1, 2, 3, 5, 10], \n    "min_samples_split": [2, 3, 5, 10],\n    "min_samples_leaf": [1, 5, 10, 20]\n}\n\nforest_model = GridSearchCV(RandomForestClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, forest_model.predict(X_train)))\nprint(forest_model.best_score_)\nprint(forest_model.best_params_)\nprint(forest_model.best_estimator_)')


# XGBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'max_depth': [3, 4, 5, 6, 7, 8], \n    'n_estimators': [5, 10, 20, 50, 100],\n    'learning_rate': np.linspace(0.02,0.16,8)\n}\n\nxgb_model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, xgb_model.predict(X_train)))\nprint(xgb_model.best_score_)\nprint(xgb_model.best_params_)\nprint(xgb_model.best_estimator_)")


# LightGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {'n_estimators': [5, 50, 100],\n              'learning_rate': np.linspace(0.02,0.16,4),\n              'num_leaves': [31, 61],\n              'min_data_in_leaf': [20, 30, 40],\n              'max_depth': range(3,8)\n}\n\nlgbm_model = GridSearchCV(lgbm.LGBMClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, lgbm_model.predict(X_train)))\nprint(lgbm_model.best_score_)\nprint(lgbm_model.best_params_)\nprint(lgbm_model.best_estimator_)")


# CatBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {'iterations': [10, 50, 100],\n              'learning_rate': np.linspace(0.02,0.12,2),\n              'depth': range(4,10)\n}\n\ncb_model = GridSearchCV(cb.CatBoostClassifier(verbose=False), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, cb_model.predict(X_train)))\nprint(cb_model.best_score_)\n# print(accuracy_score(y_val, cb_model.predict(X_val)))\nprint(cb_model.best_params_)\nprint(cb_model.best_estimator_)")


# In[ ]:


tree_train_pred = tree_model.predict(X_train)
forest_train_pred = forest_model.predict(X_train)
xgb_train_pred = xgb_model.predict(X_train)
lgbm_train_pred = lgbm_model.predict(X_train)
cb_train_pred = cb_model.predict(X_train)


# In[ ]:


submission = pd.DataFrame(
    {
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': tree_model.predict(x_test_org) 
    }
)
submission.to_csv("submission_tree.csv", index=False)

submission = pd.DataFrame(
    {
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': forest_model.predict(x_test_org)
    }
)
submission.to_csv("submission_forest.csv", index=False)

submission = pd.DataFrame(
    {
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': xgb_model.predict(x_test_org) 
    }
)
submission.to_csv("submission_xgb.csv", index=False)

submission = pd.DataFrame(
    { 
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': lgbm_model.predict(x_test_org) 
    }
)
submission.to_csv("submission_lgbm.csv", index=False)

submission = pd.DataFrame(
    { 
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': cb_model.predict(x_test_org).astype(int)
    }
)
submission.to_csv("submission_cb.csv", index=False)


# Stacking

# In[ ]:


tree_test_pred = tree_model.predict(x_test_org)
forest_test_pred = forest_model.predict(x_test_org)
xgb_test_pred = xgb_model.predict(x_test_org)
lgbm_test_pred = lgbm_model.predict(x_test_org)
cb_test_pred = cb_model.predict(x_test_org)

mean_test_pred = np.round((tree_test_pred + forest_test_pred + xgb_test_pred + lgbm_test_pred + cb_test_pred) / 5)

submission = pd.DataFrame(
    { 
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': mean_test_pred.astype(int)
    }
)
submission.to_csv("submission_mean.csv", index=False)


# In[ ]:


base_pred = pd.DataFrame({
    'tree':tree_train_pred.ravel(), 
    'forest':forest_train_pred.ravel(), 
    'xgb':xgb_train_pred.ravel(), 
    'lgbm':lgbm_train_pred.ravel(),
    'cb': cb_train_pred.ravel()
})

test_pred = pd.DataFrame({
    'tree':tree_test_pred.ravel(), 
    'forest':forest_test_pred.ravel(), 
    'xgb':xgb_test_pred.ravel(), 
    'lgbm':lgbm_test_pred.ravel(),
    'cb': cb_test_pred.ravel()
})


# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.svm import SVC\nparameters = {\n    'kernel': ['linear', 'poly', 'rbf'],\n    'C': [0.1, 0.5, 1,10,100,1000], \n    'gamma': [1, 0.1, 0.001, 0.0001, 'auto'],\n    'degree': [3, 4, 5]\n}\n\nfinal_model = GridSearchCV(SVC(), parameters, cv=5).fit(base_pred, y_train)\nprint(accuracy_score(y_train, final_model.predict(base_pred)))\nprint(final_model.best_score_)\nprint(final_model.best_params_)\nprint(final_model.best_estimator_)")


# In[ ]:


final_pred = final_model.predict(test_pred)

submission = pd.DataFrame(
    { 
        'PassengerId': titanic_test_df['PassengerId'], 
        'Survived': final_pred
    }
)
submission.to_csv("submission_final.csv", index=False)

