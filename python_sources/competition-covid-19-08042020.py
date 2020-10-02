#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from sklearn.ensemble import VotingClassifier
# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_train.csv")


# In[ ]:


train.values


# In[ ]:


y = train.covid_19.values

train = train.drop("covid_19", axis='columns')

train = train.select_dtypes(exclude=['object']).fillna(-99)

train

x = train.values

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


train_x, test_x, train_y, test_y


# In[ ]:


#VISUAL
#hists
train.hist()
plt.show()
#heat_map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


#BASIC DATA EXPLORER
train.head()
train.shape
train.columns.tolist()


# In[ ]:


model = KNeighborsClassifier(n_neighbors=100, metric="manhattan", n_jobs=-1)
model.fit(train_x, train_y)

predicted = model.predict(test_x)
predicted_p = model.predict_proba(test_x)

predicted_train = model.predict(train_x)
predicted_p_train = model.predict_proba(train_x)


# In[ ]:


for i in range(1, 201):
    model = KNeighborsClassifier(n_neighbors = i, metric = "hamming")
    model.fit(train_x, train_y)
    predicted_p = model.predict_proba(test_x) 
    predicted_p_train = model.predict_proba(train_x) 
    print(i, "-", roc_auc_score(y_score=predicted_p[:,1], y_true=test_y),
          "TRAIN:",roc_auc_score(y_score=predicted_p_train[:,1], y_true=train_y)
)

for i in range(1, 201):
    model = KNeighborsClassifier(n_neighbors = i, metric="manhattan", n_jobs=-1)
    model.fit(train_x, train_y)
    predicted_p = model.predict_proba(test_x) 
    predicted_p_train = model.predict_proba(train_x) 
    print(i, "-", roc_auc_score(y_score=predicted_p[:,1], y_true=test_y),
          "TRAIN:",roc_auc_score(y_score=predicted_p_train[:,1], y_true=train_y)
)


# In[ ]:


#BEST RESULTS
model = KNeighborsClassifier(n_neighbors = 97, metric = "hamming")
model.fit(train_x, train_y)
predicted_p = model.predict_proba(test_x) 
predicted_p_train = model.predict_proba(train_x) 
print(i, "-", roc_auc_score(y_score=predicted_p[:,1], y_true=test_y), "TRAIN:",roc_auc_score(y_score=predicted_p_train[:,1], y_true=train_y)
      
model = KNeighborsClassifier(n_neighbors = 177, metric="manhattan", n_jobs=-1)
model.fit(train_x, train_y)
predicted_p = model.predict_proba(test_x) 
predicted_p_train = model.predict_proba(train_x) 
print(i, "-", roc_auc_score(y_score=predicted_p[:,1], y_true=test_y), "TRAIN:",roc_auc_score(y_score=predicted_p_train[:,1], y_true=train_y)


# In[ ]:


xg_class = xgb.XGBClassifier(earning_rate=0.001,
                             eta = 0.6,
                             booster = 'gbtree',
                             n_estimators= 21, 
                             max_depth = 4,
                             min_child_weight = 5, 
                             gamma = 0, 
                             subsample = 1, 
                             colsample_bytree = 1,
                             objective = 'binary:logistic', 
                             nthread = 4, 
                             scale_pos_weight = 1,
                             seed = 0, 
                             silent = False)

xg_fit=xg_class.fit(train_x, train_y)

print(20, "-", roc_auc_score(test_y, xg_class.predict_proba(test_x)[:,1]))


# In[ ]:


bt = lgb.LGBMClassifier(n_jobs = -1, 
                        objective = 'binary', 
                        boosting_type = 'gbdt', 
                        learning_rate = 0.01,
                        random_state = 1,
                        subsample = 1,
                        min_split_gain = 0,
                        max_depth = 8,
                        min_child_samples = 1,
                        min_data_in_leaf = 21,
                        n_estimators = 350,
                        num_leaves = 10,
                        reg_alpha = 0.0,
                        reg_lambda = 0.1)

grid_params = {
    'n_estimators': [300, 325, 350],
    'num_leaves': [10, 11, 12],
    'min_data_in_leaf': [19, 20, 21],
    'min_child_samples': [1, 2],
    'max_depth': [7, 8, 9],
    'reg_alpha': [0.0], 
    'reg_lambda': [0.2]
}

grid = GridSearchCV(bt, cv=5, param_grid=grid_params, n_jobs=-1, scoring='roc_auc')
grid.fit(train_new_x, train_new_y)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


tree_clf = DecisionTreeClassifier(max_depth = 10, 
                                  min_samples_leaf = 20,
                                  min_samples_split = 2,
                                  max_features = 0.9, 
                                  criterion="gini", 
                                  random_state=1)  

dt = tree_clf.fit(train_new_x, train_new_y)
y_pred = dt.predict_proba(test_new_x)


# In[ ]:


param_tree_grid = {
    'max_depth': [8, 9, 10, 11, 12], 
    'min_samples_leaf': [19, 20, 21], 
    'max_features': [0.88, 0.9, 0.92],
    'min_samples_split': [1, 2, 3]
}


gridTree = GridSearchCV(tree_clf, cv=5, param_grid=param_tree_grid, n_jobs=-1, scoring='roc_auc')
gridTree.fit(train_new_x, train_new_y)


# In[ ]:


gridTree.best_score_


# In[ ]:


gridTree.best_params_


# In[ ]:


#choose importance features for the Xgboost
ImportanceFeature = pd.DataFrame({
    'variable': train.columns,
    'importance': xg_fit.feature_importances_
}).sort_values('importance', ascending=False)

ListOfImportanceFeature = list(ImportanceFeature[1:50]['variable'].values)
NewDfOnlyImportanceFeature = train[ListOfImportanceFeature]

DatasetForInsertCovid = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_train.csv")
DfWithCovid = pd.DataFrame(DatasetForInsertCovid)
NewDfOnlyImportanceFeature.insert(0,"covid_19", DfWithCovid["covid_19"])


newY = NewDfOnlyImportanceFeature.covid_19.values
train_new = NewDfOnlyImportanceFeature.drop("covid_19", axis='columns')
train_new = train_new.select_dtypes(exclude=['object']).fillna(-99)
newX = train_new.values

train_new_x, test_new_x, train_new_y, test_new_y = train_test_split(newX, newY, test_size=0.2, random_state=42)
train_new_x.shape, test_new_x.shape, train_new_y.shape, test_new_y.shape


# In[ ]:


#BEST RESULTS WITH FEATURES IMPORTANCE
for i in range(1,500):
   
    xg_class = xgb.XGBClassifier(earning_rate=0.001, 
                                 n_estimators=i, 
                                 max_depth=4,
                                 min_child_weight=5, 
                                 gamma=0, 
                                 subsample=0.8, 
                                 colsample_bytree=0.8,
                                 objective= 'binary:logistic', 
                                 nthread=4, 
                                 scale_pos_weight=1,
                                 seed=0, 
                                 silent=False)

    xg_fit=xg_class.fit(train_new_x, train_new_y)

    print(i, "-", roc_auc_score(test_new_y, xg_class.predict_proba(test_new_x)[:,1]))
    
#for i in range(1,150):
#    for j in range(1,300):
#        tree_clf = DecisionTreeClassifier(max_depth=i, 
#                                          min_samples_leaf=j,
#                                          max_features=0.9, 
#                                          criterion="gini", 
#                                          random_state=1)  
#        dt = tree_clf.fit(train_new_x, train_new_y)
#        y_pred = dt.predict_proba(test_new_x)
#        print(str(i)+"- AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_new_y-1))+ " leaf: "+str(j))


# In[ ]:



param_grid = {
    'booster': ['gbtree'], 
    'eta': [0.3, 0.4, 0.5, 0.6],
    'n_estimators': [15, 20, 25], 
    'max_depth': [1, 2, 3],    
    'min_child_weight': [4, 5, 6, 7]

}

grid = GridSearchCV(xg_class, cv=3, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
grid.fit(train_new_x, train_new_y)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


#CHOOSE THE BEST MODEL AND PARAMETERS AND PREDICT
test = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_test.csv")

TestDfOnlyImportanceFeature = test[ListOfImportanceFeature].fillna(-99)#.select_dtypes(exclude=['object']).fillna(-99)

TestDfOnlyImportanceFeature = TestDfOnlyImportanceFeature.values
TestDfOnlyImportanceFeature 
xg_class = xgb.XGBClassifier(booster = 'gbtree',
                             eta = 0.3,                             
                             earning_rate = 0.001, 
                             n_estimators = 20, 
                             max_depth = 3,
                             min_child_weight = 6, 
                             gamma = 0, 
                             subsample = 1, 
                             colsample_bytree = 1,
                             objective = 'binary:logistic', 
                             nthread = 4, 
                             scale_pos_weight = 1,
                             seed = 0, 
                             silent = False)

lgb_class = lgb.LGBMClassifier(n_jobs = -1, 
                               objective = 'binary', 
                               boosting_type = 'gbdt', 
                               learning_rate = 0.01,
                               random_state = 1,
                               subsample = 1,
                               min_split_gain = 0,
                               max_depth = 8,
                               min_child_samples = 1,
                               min_data_in_leaf = 21,
                               n_estimators = 350,
                               num_leaves = 10,
                               reg_alpha = 0.0,
                               reg_lambda = 0.1)

tree_class = DecisionTreeClassifier(max_depth = 10, 
                                    min_samples_leaf = 20,
                                    min_samples_split = 2,
                                    max_features = 0.9, 
                                    criterion="gini", 
                                    random_state=1)  

voting_classif = VotingClassifier(
    weights= [0.2, 0.6, 0.2],
    estimators=[('xg', xg_class), ('lgb', lgb_class), ('tree', tree_class)], 
    voting='soft'
) 

xg_fit = xg_class.fit(train_new_x, train_new_y)
lgb_fit = lgb_class.fit(train_new_x, train_new_y)
tree_fit = tree_class.fit(train_new_x, train_new_y)
ens_fit = voting_classif.fit(train_new_x, train_new_y) 


# In[ ]:


voting_classif = VotingClassifier(
    weights= [0.2, 0.6, 0.2],
    estimators=[('xg', xg_class), ('lgb', lgb_class), ('tree', tree_class)], 
    voting='soft'
) 

voting_cls_model = voting_classif.fit(train_new_x, train_new_y) 


# In[ ]:


for clf in (xg_class, lgb_class, tree_class, voting_cls_model):
    clf.fit(train_new_x, train_new_y)
    pred_new_y = clf.predict(test_new_x)
    print(clf.__class__.__name__, accuracy_score(test_new_y, pred_new_y))


# In[ ]:


y_pred_xg = xg_class.predict_proba(TestDfOnlyImportanceFeature)
y_pred_lgb = lgb_class.predict_proba(TestDfOnlyImportanceFeature)
y_pred_tree = tree_class.predict_proba(TestDfOnlyImportanceFeature)
y_pred_ensemble = voting_classif.predict_proba(TestDfOnlyImportanceFeature)


# In[ ]:


sub = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_submission.csv")

sub["covid_19"] = y_pred_ensemble[:, 1]

sub.to_csv("submissionOpupkin.csv", index=False)

