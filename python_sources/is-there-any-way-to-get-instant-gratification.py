#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

get_ipython().run_line_magic('matplotlib', 'inline')
# sns.set_palette(['#FF1744', '#666666'], 2)
# sns.set_style("whitegrid")

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import eli5
from eli5.sklearn import PermutationImportance


# **This is an anonymized, binary classification dataset found on a USB stick that washed ashore in a bottle.   
# There was no data dictionary with the dataset, but this poem was handwritten on an accompanying scrap of paper:**
# >> *Silly column names abound,   
# >> but the test set is a mystery.   
# >> Careful how you pick and slice,    
# >> or be left behind by history.   *

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print(train.shape, test.shape)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


# train.descirbe()


# In[ ]:


cols = []
for i in train.columns:
    cols.append(i)
cols.remove('id')
cols.remove('target')

cols_corr = []
for i in cols:
    cols_corr.append(train[[i, 'target']].corr()[i]['target'])
    
corr_df = pd.DataFrame()
corr_df['col'] = cols
corr_df['corr'] = cols_corr

plt.figure(figsize=(20, 5))
plt.bar(corr_df['col'], corr_df['corr'])
plt.xticks([])
plt.show()


# In[ ]:


train['wheezy-copper-turtle-magic'].value_counts().head()


# In[ ]:


scaler = StandardScaler()

dis_cols = [i for i in train.columns if i not in ['id', 'target', 'wheezy-copper-turtle-magic']]   
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.fit_transform(test[cols])


# In[ ]:


train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')
test['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')


# In[ ]:


train = train.sample(10000)


# In[ ]:


# train = train[['id', 'wheezy-copper-turtle-magic', 'target']]
# test = test[['id', 'wheezy-copper-turtle-magic']]


# In[ ]:


# train = pd.concat([train, pd.get_dummies(train['wheezy-copper-turtle-magic'], prefix='magic', drop_first=True)], axis=1).drop(['wheezy-copper-turtle-magic'], axis=1)
# test = pd.concat([test, pd.get_dummies(test['wheezy-copper-turtle-magic'], prefix='magic', drop_first=True)], axis=1).drop(['wheezy-copper-turtle-magic'], axis=1)


# In[ ]:


# for i in train.columns:
#     if train[i].dtype=='object':
#         print(i)


# In[ ]:


# plt.figure(figsize=(50, 50))
# sns.heatmap(train.corr(), annot=True)


# In[ ]:


X = train.drop(['id', 'target'], axis=1)
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[ ]:


model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(pred, y_test))
print(confusion_matrix(pred, y_test))


# In[ ]:


predictions = model.predict(test.drop(['id'], axis=1))
df = pd.read_csv('../input/sample_submission.csv')
df['target'] = predictions
df.to_csv('submission.csv', index=False)
df.head(10)


# In[ ]:


# model = SVC(gamma='scale')
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# model = LogisticRegression()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# model = KNeighborsClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# feature = train.columns
# importance = model.feature_importances_
# indices = np.argsort(importance)

# plt.rcParams['figure.figsize'] = (12, 50)
# plt.barh(range(len(indices)), importance[indices])
# plt.yticks(range(len(indices)), feature[indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[ ]:


# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# model = SGDClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# model = XGBClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'learning_rate': 0.05,
#     'num_leaves': 63,
#     'feature_fraction': 0.7,
# }

# lgbtrain = lgb.Dataset(X_train, label=y_train)
# gbmdl = lgb.train(params, lgbtrain, num_boost_round=20000)
# pred = gbmdl.predict(X_test,num_iteration=gbmdl.best_iteration)
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))


# In[ ]:


# type(pred, y_test)

