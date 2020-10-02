#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (9,6)})

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train_clean.csv', )
test = pd.read_csv('../input/test_clean.csv')
df = pd.concat([train, test], axis=0, sort=True)


# In[ ]:


df.head()


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

display_all(df.describe(include='all').T)


# In[ ]:


df['Survived'].value_counts()


# In[ ]:


sns.countplot(x='Pclass',data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.countplot(x='Sex',data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.countplot(x='Embarked',data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


type(df['Sex'])


# In[ ]:


df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes


# In[ ]:


categorical = ['Embarked', 'Title']

for var in categorical:
    df = pd.concat([df,pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]


# In[ ]:


df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
df.head()


# In[ ]:


train=df[pd.notnull(df['Survived'])]
X_test=df[pd.isnull(df['Survived'])].drop(['Survived'],axis=1)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(
    train.drop(['Survived'],axis=1),
    train['Survived'],
    test_size=0.2, random_state=42)


# In[ ]:


for i in [X_train, X_val, X_test]:
    print(i.shape)


# In[ ]:


rf = RandomForestClassifier(random_state=42)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


accuracy_score(y_val, rf.predict(X_val))


# In[ ]:


X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])


# In[ ]:


X_train.shape


# In[ ]:


rf = RandomForestClassifier(n_estimators=10, random_state=42)
cross_val_score(rf, X_train, y_train, cv=5)


# In[ ]:


cross_val_score(rf, X_train, y_train, cv=5).mean()


# In[ ]:


help(RandomForestClassifier)


# In[ ]:


n_estimators = [10, 100, 1000, 2000]
max_depth = [None, 5, 10, 20]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)


# In[ ]:


# create the default model
rf = RandomForestClassifier(random_state=42)

# search the grid
grid = GridSearchCV(estimator=rf, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2,
                    n_jobs=-1)

grid_result = grid.fit(X_train, y_train)


# In[ ]:


grid_result.best_estimator_


# In[ ]:


grid_result.best_params_


# In[ ]:


grid_result.best_score_


# In[ ]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


# create the grid
leaf_samples = [1, 2, 3, 4, 5, 6]
param_grid = dict(min_samples_leaf=leaf_samples)

# create the model with new max_depth and n_estimators
rf = grid_result.best_estimator_

# search the grid
grid = GridSearchCV(estimator=rf, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2,
                    n_jobs=-1)

grid_result = grid.fit(X_train, y_train)


# In[ ]:





# In[ ]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


# create the grid
max_features = [5, 8, 10, 12, None]
bootstrap = [True, False]
param_grid = dict(max_features=max_features, bootstrap=bootstrap)

# create the model with new leaf size
rf = grid_result.best_estimator_

# search the grid
grid = GridSearchCV(estimator=rf, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2,
                    n_jobs=-1)

grid_result = grid.fit(X_train, y_train)


# In[ ]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


rf=grid_result.best_estimator_


# In[ ]:


cross_val_score(rf, X_train, y_train, cv=5).mean()


# In[ ]:


test['Survived'] = rf.predict(X_test)


# In[ ]:


solution = test[['PassengerId', 'Survived']]
solution['Survived'] = solution['Survived'].apply(int)


# In[ ]:


solution.head(10)


# In[ ]:


solution.to_csv("Submit_titanic.csv",index=False)


# In[ ]:




