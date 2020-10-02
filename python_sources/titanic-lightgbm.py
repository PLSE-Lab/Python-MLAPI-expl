#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.metrics import accuracy_score

RANDOM_STATE = 17


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_all = pd.concat([train, test], sort=True).reset_index(drop=True)
df_all.head()


# In[ ]:


def divide_df(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)


# ## 1. Data Analysis

# In[ ]:


print(train.info())


# In[ ]:


print(test.info())


# ### 1.1 Missing Values

# In[ ]:


def display_missing(data, name):
    missing_data = data.isna().sum()
    print(name)
    for feature, i in missing_data.items():
        print('%s column missing values: %s' % (feature, i))
    print('\n')



display_missing(train, 'Train')
display_missing(test, 'Test')


# **1.1.1. Age**
# 
# Missing values in Age are filled with median age of corresponding Pclass group, since Pclass has high correlation with Age

# In[ ]:


numerical = list(set(df_all.columns) - 
                 set(['Sex', 'Embarked', 'Survived']))

corr_matrix = df_all[numerical].corr()
fig, ax = plt.subplots(1,3,figsize=(15,4))
sns.heatmap(corr_matrix, annot=True,  ax=ax[0], fmt=".2f");
sns.boxplot(x='Sex', y='Age', data=df_all, ax=ax[1]);
sns.boxplot(x='Embarked', y='Age', data=df_all, ax=ax[2]);
fig.show()


# In[ ]:


df_all['Age'] = df_all.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# **1.1.2. Fare**
# 
# Missing values in Fare are filled with median age of corresponding Pclass group, since Pclass has high correlation with Fare

# In[ ]:


df_all['Fare'] = df_all.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))


# **1.1.3. Embarked**

# In[ ]:


embarked_mode = df_all['Embarked'].mode()[0]
df_all['Embarked'] = df_all['Embarked'].fillna(embarked_mode)
print('Missing values of Embarked filled with:', embarked_mode)


# **1.1.4 Cabin**
# 
# Cabin has many missing values. Let's just drop it

# In[ ]:


df_all.drop('Cabin', axis=1, inplace=True)


# ### 1.2 Target distribution

# In[ ]:


fig1, ax1 = plt.subplots()
ax1.pie(train['Survived'].groupby(train['Survived']).count(), 
        labels = ['Not Survived', 'Survived'], autopct = '%1.1f%%')
ax1.axis('equal')

plt.show()


# ### 1.3 Feature engineering

# In[ ]:


df_all.head()


# **1.3.1 Name**
# 
# Let's extract useful information from Name

# In[ ]:


df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
fig, ax = plt.subplots(figsize=(15,5))
sns.countplot(x='Title', hue='Survived', data=df_all, ax=ax);
fig.show()


# Group Title into 4 categories

# In[ ]:


df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 
                                           'Mlle', 'Lady', 'Mme', 
                                           'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 
                                           'Capt', 'Sir', 'Don', 'Rev'], 'Other')
sns.countplot(x='Title', hue='Survived', data=df_all);


# In[ ]:


df_all.drop('Name', axis=1, inplace=True)


# **1.3.2 Ticket**
# 
# Just drop this column

# In[ ]:


df_all.drop('Ticket', axis=1, inplace=True)


# In[ ]:


df_all.head()


# ### 1.4 Encode categorical features

# In[ ]:


sex_dict = {'male':1, 'female':0}
df_all['Sex'] = df_all['Sex'].map(sex_dict)


# In[ ]:


embarked_dict = {'S':0, 'Q':1, 'C':2}
df_all['Embarked'] = df_all['Embarked'].map(embarked_dict)


# In[ ]:


title_dict = {'Mr':0, 'Miss/Mrs/Ms':1, 'Master':2, 'Other':3}
df_all['Title'] = df_all['Title'].map(title_dict)


# In[ ]:


df_all.head()


# ## 2. Model

# In[ ]:


df_train, df_test = divide_df(df_all)
df_train = df_train.drop('PassengerId', axis=1)
df_train.head()


# In[ ]:


X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_STATE)

categorical_features = ['Embarked', 'Pclass', 'Title']


# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# ### 2.1 Logistic Regression

# In[ ]:


logit = LogisticRegression(solver='lbfgs', max_iter=500,
                           random_state=RANDOM_STATE, n_jobs=-1)

parameters = {'C':[0.005, 0.01, 0.05, 0.1, 0.5, 1]}
logit_grid = GridSearchCV(logit, parameters)

logit_pipe = Pipeline([('preprocessor', preprocessor),
                       ('scaler', StandardScaler()), 
                       ('logit', logit_grid)])
logit_pipe.fit(X_train, y_train);


# In[ ]:


print('Best C: ', logit_grid.best_params_['C'])


# In[ ]:


print('Train accuracy:', accuracy_score(y_train, logit_pipe.predict(X_train)))
print('CV accuracy:', logit_grid.best_score_)
print('Test accuracy:', accuracy_score(y_valid, logit_pipe.predict(X_valid)))


# ### 2.2 Random forest

# In[ ]:


rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

parameters = {'n_estimators':[3, 5, 10, 30],
              'max_depth': range(1, 10)}

rf_grid = GridSearchCV(rf, parameters)

rf_pipe = Pipeline([('preprocessor', preprocessor), 
                    ('forest', rf_grid)])
rf_pipe.fit(X_train, y_train);


# In[ ]:


print('Best n_estimators: ', rf_grid.best_params_['n_estimators'])
print('Best max_depth: ', rf_grid.best_params_['max_depth'])


# In[ ]:


print('Train accuracy:', accuracy_score(y_train, rf_pipe.predict(X_train)))
print('CV accuracy:', rf_grid.best_score_)
print('Test accuracy:', accuracy_score(y_valid, rf_pipe.predict(X_valid)))


# ### 2.3 LightGBM

# In[ ]:


lgb_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)


# In[ ]:


#lgb_clf.fit(X_train, y_train)
#accuracy_score(y_valid, lgb_clf.predict(X_valid))


# **2.3.1 First stage of hyper-param tuning: tuning model complexity**

# In[ ]:


param_grid = {'num_leaves': [7, 15, 31, 63], 
              'max_depth': [1, 2, 3, 4, 5, 6, -1]}

lgb_grid = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, 
                             cv=5, verbose=1, n_jobs=-1)

lgb_grid.fit(X_train, y_train, categorical_feature=categorical_features);


# In[ ]:


print('Best params: ', lgb_grid.best_params_)


# In[ ]:


print('Train accuracy:', accuracy_score(y_train, lgb_grid.predict(X_train)))
print('CV accuracy:', lgb_grid.best_score_)
print('Test accuracy:', accuracy_score(y_valid, lgb_grid.predict(X_valid)))


# **2.3.2 Second stage of hyper-param tuning: convergence:**

# In[ ]:


num_iterations = 2000
lgb_clf2 = lgb.LGBMClassifier(random_state=RANDOM_STATE, 
                              max_depth=lgb_grid.best_params_['max_depth'], 
                              num_leaves=lgb_grid.best_params_['num_leaves'], 
                              n_estimators=num_iterations,
                              n_jobs=-1)

param_grid2 = {'learning_rate': np.logspace(-4, 0, 10)}
lgb_grid2 = GridSearchCV(estimator=lgb_clf2, param_grid=param_grid2,
                               cv=5, verbose=1, n_jobs=4)

lgb_grid2.fit(X_train, y_train, categorical_feature=categorical_features)


# In[ ]:


print('Best params: ', lgb_grid2.best_params_)


# In[ ]:


print('Train accuracy:', accuracy_score(y_train, lgb_grid2.predict(X_train)))
print('CV accuracy:', lgb_grid2.best_score_)
print('Test accuracy:', accuracy_score(y_valid, lgb_grid2.predict(X_valid)))


# In[ ]:


final_lgb = lgb.LGBMClassifier(n_estimators=num_iterations,
                               max_depth=lgb_grid.best_params_['max_depth'], 
                               num_leaves=lgb_grid.best_params_['num_leaves'],
                               learning_rate=lgb_grid2.best_params_['learning_rate'],
                               n_jobs=-1, random_state=RANDOM_STATE)


# In[ ]:


final_lgb.fit(X, y, categorical_feature=categorical_features)


# In[ ]:


pd.DataFrame(final_lgb.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]


# ## 3. **Submission**

# In[ ]:


ids = df_test['PassengerId'].values
test_inputs = df_test.drop('PassengerId', axis=1)


# In[ ]:


predsTest = final_lgb.predict(test_inputs)
y = np.int32(predsTest > 0.5)
y = y.astype(int)

output = pd.DataFrame({'PassengerId': ids, 'Survived': y})
output.to_csv("submission.csv", index=False)

