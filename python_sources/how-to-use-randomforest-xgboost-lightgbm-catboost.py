#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgbm
import catboost as cb

np.random.seed(42)


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# In[ ]:


used_columns = ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']


# In[ ]:


y_train = train_df[['Survived']]
y_train.shape


# In[ ]:


X_all = train_df[used_columns].append(test_df[used_columns])
X_all.shape


# In[ ]:


X_all.isna().sum()


# In[ ]:


X_all['Embarked'].value_counts()


# In[ ]:


X_all['Embarked'].fillna('S', inplace=True)
X_all['Fare'].fillna(X_all['Fare'].median(), inplace=True)


# In[ ]:


X_all['Title'] = X_all['Name'].str.extract(' ([A-Za-z]+)\.')
X_all['Title'] = X_all['Title'].replace(['Ms', 'Mlle'], 'Miss')
X_all['Title'] = X_all['Title'].replace(['Mme', 'Countess', 'Lady', 'Dona'], 'Mrs')
X_all['Title'] = X_all['Title'].replace(['Dr', 'Major', 'Col', 'Sir', 'Rev', 'Jonkheer', 'Capt', 'Don'], 'Mr')


# In[ ]:


# X_all = pd.concat([X_all, pd.get_dummies(X_all[['Sex', 'Embarked']])], axis=1)
X_all["Sex"] = X_all["Sex"].map({"male": 1, "female": 0}).astype(int)    
X_all["Embarked"] = X_all["Embarked"].map({"S": 1, "C": 2, "Q": 3}).astype(int)    
X_all['Title'] = X_all['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}).astype(int)   


# In[ ]:


X_all['TicketNumber'] = X_all['Ticket'].str.split()
X_all['TicketNumber'] = X_all['TicketNumber'].str[-1]
X_all['TicketNumber'] = LabelEncoder().fit_transform(X_all['TicketNumber'])


# In[ ]:


X_all.head()


# In[ ]:


X_all.drop(['Name', 'Ticket'], axis=1, inplace=True)
X_all.head()


# In[ ]:


X_all['FamilySize'] = X_all['SibSp'] + X_all['Parch'] + 1
X_all['IsAlone'] = X_all['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
# X_all['SquaredFare'] = X_all['Fare'] ** 2
X_all.head()


# In[ ]:


categorical_columns = ['Sex', 'Parch', 'Embarked', 'Title', 'TicketNumber', 'IsAlone']


# In[ ]:


X_train = X_all[0:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[ ]:


y_train = np.ravel(y_train)
# y_val = np.ravel(y_val)


# ### Decision tree

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "criterion": ["gini", "entropy"],\n    "max_depth": [1, 2, 3, 5, 10, None], \n    "min_samples_split": [2, 3, 5, 10],\n    "min_samples_leaf": [1, 5, 10, 20]\n}\n\ntree_model = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, tree_model.predict(X_train)))\nprint(tree_model.best_score_)\n# print(accuracy_score(y_val, tree_model.predict(X_val)))\nprint(tree_model.best_params_)\nprint(tree_model.best_estimator_)')


# ### Random forest

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "n_estimators": [2, 4, 5, 8, 10, 15], \n    "criterion": ["gini", "entropy"],\n    "max_features": ["auto", "log2"], \n    "max_depth": [1, 2, 3, 5, 10], \n    "min_samples_split": [2, 3, 5, 10],\n    "min_samples_leaf": [1, 5, 10, 20]\n}\n\nforest_model = GridSearchCV(RandomForestClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, forest_model.predict(X_train)))\nprint(forest_model.best_score_)\n# print(accuracy_score(y_val, forest_model.predict(X_val)))\nprint(forest_model.best_params_)\nprint(forest_model.best_estimator_)')


# ### XGBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'max_depth': [3, 4, 5, 6, 7, 8], \n    'n_estimators': [5, 10, 20, 50, 100],\n    'learning_rate': np.linspace(0.02,0.16,8)\n}\n\nxgb_model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, xgb_model.predict(X_train)))\nprint(xgb_model.best_score_)\n# print(accuracy_score(y_val, xgb_model.predict(X_val)))\nprint(xgb_model.best_params_)\nprint(xgb_model.best_estimator_)")


# ### LightGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {'n_estimators': [5, 50, 100],\n              'learning_rate': np.linspace(0.02,0.16,4),\n              'num_leaves': [31, 61],\n              'min_data_in_leaf': [20, 30, 40],\n              'max_depth': range(3,8),\n}\n\nlgbm_model = GridSearchCV(lgbm.LGBMClassifier(), parameters, cv=5).fit(X_train, y_train, categorical_feature=categorical_columns)\nprint(accuracy_score(y_train, lgbm_model.predict(X_train)))\nprint(lgbm_model.best_score_)\n# print(accuracy_score(y_val, lgbm_model.predict(X_val)))\nprint(lgbm_model.best_params_)\nprint(lgbm_model.best_estimator_)")


# ### CatBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {'iterations': [10, 50, 100],\n              'learning_rate': np.linspace(0.02,0.16,4),\n              'depth': range(4,10)\n}\n\ncb_model = GridSearchCV(cb.CatBoostClassifier(verbose=False), parameters, cv=5).fit(X_train, y_train)\nprint(accuracy_score(y_train, cb_model.predict(X_train)))\nprint(cb_model.best_score_)\n# print(accuracy_score(y_val, cb_model.predict(X_val)))\nprint(cb_model.best_params_)\nprint(cb_model.best_estimator_)")


# In[ ]:


submission = pd.DataFrame(
    {
        'PassengerId': test_df['PassengerId'], 
        'Survived': tree_model.predict(X_test) 
    }
)
submission.to_csv("submission_tree.csv", index=False)

submission = pd.DataFrame(
    {
        'PassengerId': test_df['PassengerId'], 
        'Survived': forest_model.predict(X_test)
    }
)
submission.to_csv("submission_forest.csv", index=False)

submission = pd.DataFrame(
    {
        'PassengerId': test_df['PassengerId'], 
        'Survived': xgb_model.predict(X_test) 
    }
)
submission.to_csv("submission_xgb.csv", index=False)

submission = pd.DataFrame(
    { 
        'PassengerId': test_df['PassengerId'], 
        'Survived': lgbm_model.predict(X_test) 
    }
)
submission.to_csv("submission_lgbm.csv", index=False)

submission = pd.DataFrame(
    { 
        'PassengerId': test_df['PassengerId'], 
        'Survived': cb_model.predict(X_test).astype(int)
    }
)
submission.to_csv("submission_cb.csv", index=False)


# ### Stacking

# In[ ]:


tree_test_pred = tree_model.predict(X_test)
forest_test_pred = forest_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)
lgbm_test_pred = lgbm_model.predict(X_test)
cb_test_pred = cb_model.predict(X_test)

mean_test_pred = np.round((tree_test_pred + forest_test_pred + xgb_test_pred + lgbm_test_pred + cb_test_pred) / 5)

submission = pd.DataFrame(
    { 
        'PassengerId': test_df['PassengerId'], 
        'Survived': mean_test_pred.astype(int)
    }
)
submission.to_csv("submission_mean.csv", index=False)


# In[ ]:


tree_train_pred = tree_model.predict(X_train)
forest_train_pred = forest_model.predict(X_train)
xgb_train_pred = xgb_model.predict(X_train)
lgbm_train_pred = lgbm_model.predict(X_train)
cb_train_pred = cb_model.predict(X_train)


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


get_ipython().run_cell_magic('time', '', "from sklearn.svm import SVC\nparameters = {\n    'kernel': ['linear', 'poly', 'rbf'],\n    'C': [0.1, 0.5, 1,10,100,1000], \n    'gamma': [1, 0.1, 0.001, 0.0001, 'auto'],\n    'degree': [3, 4, 5]\n}\n\nfinal_model = GridSearchCV(SVC(), parameters, cv=5).fit(base_pred, y_train)\nprint(accuracy_score(y_train, final_model.predict(base_pred)))\nprint(final_model.best_score_)\n# print(accuracy_score(y_val, xgb_model.predict(X_val)))\nprint(final_model.best_params_)\nprint(final_model.best_estimator_)")


# In[ ]:


final_pred = final_model.predict(test_pred)

submission = pd.DataFrame(
    { 
        'PassengerId': test_df['PassengerId'], 
        'Survived': final_pred
    }
)
submission.to_csv("submission_final.csv", index=False)


# In[ ]:




