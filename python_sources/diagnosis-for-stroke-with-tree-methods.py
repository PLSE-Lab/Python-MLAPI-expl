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


# pd.read_csv('../input/test_2v.csv')
address = "../input/train_2v.csv"
df = pd.read_csv(address)
to_drop = ['id']
df = df.drop(to_drop, axis=1).dropna()
df = df.loc[df['gender']!='Other']
df.head()


# In[ ]:


for cat in df.select_dtypes('O').columns:
    df[cat] = df[cat].astype('category').cat.codes
df.head()


# In[ ]:


import seaborn as sns
sns.countplot("stroke",data=df)


# In[ ]:


count_class_0, count_class_1 = df['stroke'].value_counts()

# Divide by class
df_class_0 = df[df['stroke'] == 0]
df_class_1 = df[df['stroke'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_under = pd.concat([df_class_0_under, df_class_1], axis=0)
print(df_under['stroke'].value_counts())
df = df_under


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('stroke',axis=1),df['stroke'])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train,y_train)

rf.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(min_samples_leaf=10, n_estimators=100).fit(X_train,y_train)
# from sklearn.model_selection import GridSearchCV
# param = {'n_estimators':[500,200,100],'min_samples_leaf':[10,20,50]}
# grid = GridSearchCV(clf,param).fit(X_train,y_train)
# print(grid.best_params_)
rf_pred = rf.predict(X_test)
print(rf.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,rf_pred))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier().fit(X_train,y_train)
gbdt_pred = gbdt.predict(X_test)
gbdt_prob = gbdt.predict_proba(X_test)


# In[ ]:


from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
dummy_majority.score(X_test,y_test)


# In[ ]:


rf_pred = rf.predict(X_test)
dummy_majority_pred = dummy_majority.predict(X_test)
print(confusion_matrix(y_test,rf_pred))
print(confusion_matrix(y_test,dummy_majority_pred))


# In[ ]:


from sklearn.metrics import accuracy_score,recall_score, roc_auc_score, confusion_matrix
print(accuracy_score(y_test,rf_pred))
print(recall_score(y_test,rf_pred))
# print(roc_auc_score(y_test,pred))
print(confusion_matrix(y_test,rf_pred))


# In[ ]:


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
col_name = df.drop('stroke',axis=1).columns.values
# Plot the feature importances of the forest
import matplotlib.pyplot as plt
fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]),
        importances[indices],
        yerr=std[indices],
        align="center")
plt.xticks(range(X_train.shape[1]), np.array(col_name)[indices])
plt.xlim([-1, X_train.shape[1]])
fig.autofmt_xdate()
plt.show()


# In[ ]:


importances = gbdt.feature_importances_
indices = np.argsort(importances)[::-1]
col_name = df.drop('stroke',axis=1).columns.values
# Plot the feature importances of the forest
import matplotlib.pyplot as plt
fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align="center")
plt.xticks(range(X_train.shape[1]), np.array(col_name)[indices])
plt.xlim([-1, X_train.shape[1]])
fig.autofmt_xdate()
plt.show()


# In[ ]:


import graphviz
from sklearn import tree
dt = tree.DecisionTreeClassifier(min_samples_leaf=30).fit(X_train,y_train)
dt_pred = dt.predict(X_test)

feature_names = df.drop('stroke',axis=1).columns.values
target_names = ['well','stroke']

dot_data = tree.export_graphviz(dt, out_file=None, 
                     feature_names=feature_names,  
                     class_names=target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)
graph.render("dt_stroke") 


# In[ ]:




