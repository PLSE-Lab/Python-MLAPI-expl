#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import os
get_ipython().run_line_magic('matplotlib', 'inline')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dtrain = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip')
dtest = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip')


# ## Data Preprocessing

# In[ ]:


dtrain.head()


# In[ ]:


dtrain.info()


# In[ ]:


dtrain.shape


# In[ ]:


dtrain.describe()


# ## Exploratory Data Analysis

# First, let's check the distribution

# In[ ]:


f,ax = plt.subplots(1,4, figsize=(15,3))

sns.distplot(dtrain['bone_length'], ax = ax[0])
sns.distplot(dtrain['rotting_flesh'], ax = ax[1], color = 'r')
sns.distplot(dtrain['hair_length'], ax = ax[2], color = 'g')
sns.distplot(dtrain['has_soul'], ax = ax[3], color = 'purple')

plt.show()


# Seems like their distribution is quiet normal. Let see if we use log transformation to our data.

# In[ ]:


f,ax = plt.subplots(1,4, figsize=(15,3))

sns.distplot(np.log(dtrain['bone_length']), ax = ax[0])
sns.distplot(np.log(dtrain['rotting_flesh']), ax = ax[1], color = 'r')
sns.distplot(np.log(dtrain['hair_length']), ax = ax[2], color = 'g')
sns.distplot(np.log(dtrain['has_soul']), ax = ax[3], color = 'purple')

plt.show()


# The distribution has negative skewness then. Now let's plot them with scatter plot.

# In[ ]:


sns.pairplot(data=dtrain.iloc[:,1:], hue = 'type')


# What about their correlation?

# In[ ]:


sns.heatmap(dtrain.drop('id', axis = 1).corr(), square = True, annot = True)


# From the above, we can get some information:
# * `hair_length` and `has_soul` have a good positive correlation
# * We also know that `rotting_flesh` has negative correlation with all other features

# ## Feature Engineering

# We do label encoding to make the ML learn easier.

# In[ ]:


dtrain['type'].unique()


# In[ ]:


label_encoder = LabelEncoder()
dtrain['type'] = label_encoder.fit_transform(dtrain['type'])
# 0 : Ghost
# 1 : Ghoul
# 2 : Goblin


# In[ ]:


dtrain['type'].unique()


# The `type` feature is in number now. So it's time to encode the `color` feature.

# In[ ]:


dtrain['color'].unique()


# In[ ]:


d_color = dtrain['color']


# In[ ]:


dtrain['clear'] = [1 if d_color[i] == 'clear' else 0 for i in range(len(d_color))]
dtrain['green'] = [1 if d_color[i] == 'green' else 0 for i in range(len(d_color))]
dtrain['black'] = [1 if d_color[i] == 'black' else 0 for i in range(len(d_color))]
dtrain['white'] = [1 if d_color[i] == 'white' else 0 for i in range(len(d_color))]
dtrain['blue'] = [1 if d_color[i] == 'blue' else 0 for i in range(len(d_color))]
dtrain['blood'] = [1 if d_color[i] == 'blood' else 0 for i in range(len(d_color))]


# In[ ]:


d_color2 = dtest['color']


# In[ ]:


dtest['clear'] = [1 if d_color2[i] == 'clear' else 0 for i in range(len(d_color2))]
dtest['green'] = [1 if d_color2[i] == 'green' else 0 for i in range(len(d_color2))]
dtest['black'] = [1 if d_color2[i] == 'black' else 0 for i in range(len(d_color2))]
dtest['white'] = [1 if d_color2[i] == 'white' else 0 for i in range(len(d_color2))]
dtest['blue'] = [1 if d_color2[i] == 'blue' else 0 for i in range(len(d_color2))]
dtest['blood'] = [1 if d_color2[i] == 'blood' else 0 for i in range(len(d_color2))]


# In[ ]:


dtrain.head()


# In[ ]:


dtest.head()


# Now it's time to build our model

# ## Modelling

# In[ ]:


train = dtrain.copy()
y = train['type']
x = train.drop(['id', 'color','type'], axis = 1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 0) 
fold = KFold(n_splits = 5)


# In[ ]:


scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}


# In[ ]:


ensembles=[]
ensembles.append(('rfc',RandomForestClassifier(n_estimators=10)))
ensembles.append(('abc',AdaBoostClassifier(n_estimators=10)))
ensembles.append(('bc',BaggingClassifier(n_estimators=10)))
ensembles.append(('etc',ExtraTreesClassifier(n_estimators=10)))

results=[]
names=[]
for name,model in ensembles:
    result = cross_val_score(model,x_train,y_train,cv=fold,scoring='accuracy')
    results.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)


# In[ ]:


# Random Forest Tuning
n_estimators=[10,20,30,40,50]
max_depth =  [4,6,8,10,12,24]

param_grid=dict(n_estimators=n_estimators, max_depth=max_depth)

model=RandomForestClassifier()

fold=KFold(n_splits=10,random_state=0)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)
grid_result=grid.fit(x_train,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


rf_best_params = grid_result.best_params_


# In[ ]:


# AdaBoost Tuning
n_estimators=[10,20,30,40,50]
learning_rate =  [1.0, 0.1, 0.05, 0.01, 0.001]
param_grid=dict(n_estimators=n_estimators, learning_rate=learning_rate)


model=AdaBoostClassifier()

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)
grid_result=grid.fit(x_train,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


ab_best_params = grid_result.best_params_


# In[ ]:


# Bagging Tuning
n_estimators=[10,20,30,40,50]
max_features =  [2,4,6,8,10]

param_grid=dict(n_estimators=n_estimators, max_features=max_features)

model=BaggingClassifier()

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)
grid_result=grid.fit(x_train,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


bc_best_params = grid_result.best_params_


# In[ ]:


# Extra Trees Tuning
n_estimators=[10,20,30,40,50]
max_depth =  [4,6,8,10,12,24]

param_grid=dict(n_estimators=n_estimators, max_depth=max_depth)

model=ExtraTreesClassifier()

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)
grid_result=grid.fit(x_train,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


et_best_params = grid_result.best_params_


# based on our tuning, Let's use VotingClassifier.

# In[ ]:


rf = RandomForestClassifier(**rf_best_params)
et = ExtraTreesClassifier(**et_best_params)
bc = BaggingClassifier(**bc_best_params)
ab = AdaBoostClassifier(rf, **ab_best_params)

vc = VotingClassifier(estimators=[('et', et), ('rf', rf), ('bc', bc), ('ab', ab)], voting='soft')


# In[ ]:


vc.fit(x_train, y_train)
y_pred = vc.predict(x_test)
vc_acc = accuracy_score(y_pred, y_test)
print("Accuracy score: ", vc_acc)


# In[ ]:


# It's time to predict the data
y_pred = vc.predict(dtest.drop(['id','color'], axis = 1))
y_pred


# In[ ]:


sub = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip')
sub['type'] = y_pred
sub['type'] = sub['type'].map({
    0:'Ghost',
    1:'Ghoul',
    2:'Goblin'
})


# In[ ]:


sub


# In[ ]:


# submit
sub.to_csv('submission.csv', index = False)

