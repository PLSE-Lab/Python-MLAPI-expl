#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

sns.set()


# In[ ]:



df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)


df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)


# In[ ]:


df_train.head()


# In[ ]:


df_test['target'] = np.nan
df = pd.concat([df_train, df_test])


# In[ ]:


#Data analyzing
Numeric_features = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week', 'target'
]
Categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationshop',
    'race', 'sex', 'native-country'
]


# In[ ]:


sns.countplot(df['target'], label="Count")


# In[ ]:


#Check correlation
plt.figure(figsize=(12, 4))
sns.heatmap(df[Numeric_features].corr(), annot=True, cmap='Greens')


# In[ ]:


graphs = sns.FacetGrid(df, col='target')
graphs = graphs.map(sns.distplot, 'age')


# In[ ]:


df_tmp = df.loc[df['target'].notna()].groupby(['education'])['target'].agg(
    ['mean', 'std']).rename(columns={
        'mean': 'target_mean',
        'std': 'target_std'
    }).fillna(0.0).reset_index()


# In[ ]:


df = pd.merge(df, df_tmp, how='left', on=['education'])


# In[ ]:





# In[ ]:


#Feature Enginering
df.head()


# In[ ]:


df['sex'].unique()


# In[ ]:


df['race'].unique()


# In[ ]:


df['sex'] = df['sex'].replace(' Male', 0)
df['sex'] = df['sex'].replace(' Female', 1)


# In[ ]:


married = [i for i in df['marital-status'].unique() if i[:8] == ' Married']
alone = [i for i in df['marital-status'].unique() if i not in married]


# In[ ]:


df['marital-status'] = df['marital-status'].replace(married, 1)
df['marital-status'] = df['marital-status'].replace(alone, 0)


# In[ ]:


df['marital-status'].unique()


# In[ ]:


df.head()


# In[ ]:


df.drop(columns=[
    'uid', 'workclass', 'occupation','education', 'relationship', 'race', 'native-country'
],
        inplace=True)


# In[ ]:


df.head()


# In[ ]:


our_x_train = df.loc[df['target'].notna()].drop(columns=['target'])
our_y_train = df.loc[df['target'].notna()]['target']
our_x_test = df.loc[df['target'].isna()].drop(columns=['target'])
our_y_test = df.loc[df['target'].isna()]['target']


# Cross - Validation
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(our_x_train,
                                                    our_y_train,
                                                    test_size=0.33,
                                                    random_state=17)


# In[ ]:


plt.figure(figsize=(15, 10))

#critetion
plt.subplot(3, 3, 1)
feature_param = ['gini', 'entropy']
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(criterion=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Criterion')
plt.grid()

#max_depth
plt.subplot(3, 3, 2)
max_depth_check = range(1, 30)
scores = []
for depth in max_depth_check:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(max_depth_check, scores, '.-')
plt.axis('tight')
plt.title('Depth')
plt.grid()

#Splitter
plt.subplot(3, 3, 3)
feature_param = ['best', 'random']
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(splitter=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Splitter')
plt.grid()

#Min Samples Leaf
plt.subplot(3, 3, 4)
feature_param = range(2, 21)
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(min_samples_leaf=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Min Samples Leaf')
plt.grid()

#Min Samples Split
plt.subplot(3, 3, 5)
feature_param = range(2, 21)
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(min_samples_split=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Min Samples Split')
plt.grid()

#max_features
plt.subplot(3, 3, 6)
feature_param = range(1, df.shape[1])
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(max_features=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Max Features')
plt.grid()


# In[ ]:


model = DecisionTreeClassifier(criterion='gini',
                               splitter='best',
                               max_depth=10,
                               max_features=5,
                               min_samples_split=40,
                               min_samples_leaf=12)
parameter_grid = {
    'max_depth': range(5, 15),
    'max_features': range(1, 9),
    'min_samples_split': [35, 40, 45, 50],
    'min_samples_leaf': [5, 10, 15, 20],
}
grid_search = GridSearchCV(model, param_grid=parameter_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[ ]:


print(f'Best score: {grid_search.best_score_}')
print(f'Best parameters: {grid_search.best_params_}')
print(accuracy_score(y_test, grid_search.predict(X_test)))


# In[ ]:


tree_cl = DecisionTreeClassifier(criterion='gini',
                               splitter='best',
                               max_depth=8,
                               max_features=6,
                               min_samples_split=45,
                               min_samples_leaf=10)
tree_cl.fit(X_train,y_train)
tree_cl.score(X_test,y_test)


# In[ ]:


tree_cl.predict(our_x_test)


# In[ ]:


p = tree_cl.predict_proba(our_x_test)[:,1]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")


# In[ ]:


sns.distplot(p)


# In[ ]:


df_submit = pd.DataFrame({
    'uid': df_test['uid'],
    'target': p
})


# In[ ]:


df_submit.to_csv('submit.csv', index=False)


# In[ ]:




