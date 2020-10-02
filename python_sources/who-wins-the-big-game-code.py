#!/usr/bin/env python
# coding: utf-8

# ## HackerEarth Machine Learning Competition.
# This is a note book which is made as a solution for HackerEarth Ml competition['Who wins the Big Game']

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
""
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/who-wins-the-big-game/sample_submission.csv")
test = pd.read_csv("../input/who-wins-the-big-game/test.csv")
train = pd.read_csv("../input/who-wins-the-big-game/train.csv")


# In[ ]:


train.head(10)


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


new_train=train.loc[:, train.columns != 'ID']


# In[ ]:


new_train['Team_Value'].unique()


# In[ ]:


class_map = {
    'Less_Than_Four_Billion': 0,
    'Above_Four_Billion': 1,
    'Less_Than_Three_Billion': 2,
}
new_train['Team_Value'] = new_train['Team_Value'].map(class_map)


# In[ ]:


new_train['Playing_Style'].unique()


# In[ ]:


class_map_playing={
    'Balanced':0,
    'Aggressive_Offense':1,
    'Aggressive_Defense':2,
    'Relaxed':3
}
new_train['Playing_Style']=new_train['Playing_Style'].map(class_map_playing)


# In[ ]:


new_train['Number_Of_Injured_Players'].unique()


# In[ ]:


injured_Players_Map ={
    'five':5,
    'four':4,
    'six':6,
    'three':3,
    'seven':7,
    'eight':8,
    'two':2,
    'nine':9,
    'one':1,
    'ten':10
}
new_train['Number_Of_Injured_Players'] =new_train['Number_Of_Injured_Players'].map(injured_Players_Map)


# In[ ]:


new_train['Coach_Experience_Level'].unique()


# In[ ]:


Coach_Experience_Level_map={
    'Intermediate':1,
    'Beginner':0,
    'Advanced':2
    
}
new_train['Coach_Experience_Level'] = new_train['Coach_Experience_Level'].map(Coach_Experience_Level_map)


# In[ ]:


new_train['Coach_Experience_Level'].unique()


# In[ ]:


new_train_target = new_train['Won_Championship']
new_train = new_train.drop('Won_Championship',axis=1)
new_train.head()


# In[ ]:


new_test=test.drop('ID',axis=1)


# In[ ]:


new_test['Team_Value'].unique()
new_test['Team_Value'] = new_test['Team_Value'].map(class_map)


# In[ ]:


new_test['Playing_Style'].unique()
new_test['Playing_Style']=new_test['Playing_Style'].map(class_map_playing)


# In[ ]:


new_test['Number_Of_Injured_Players'].unique()


# In[ ]:


new_test['Number_Of_Injured_Players'] =new_test['Number_Of_Injured_Players'].map(injured_Players_Map)


# In[ ]:


new_test['Coach_Experience_Level'].unique()


# In[ ]:


new_test['Coach_Experience_Level'] = new_test['Coach_Experience_Level'].map(Coach_Experience_Level_map)


# In[ ]:


new_train.info()


# In[ ]:


new_train.describe()


# In[ ]:


print(new_train_target.describe())
train_target=new_train_target


# In[ ]:


new_test.info()


# In[ ]:


new_test.describe()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
RandomForestRegressor()


# In[ ]:


new_train.values


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(new_train)


# In[ ]:


best_svr = SVR(kernel='linear')


# In[ ]:


new_train


# In[ ]:


from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(best_svr, new_train, train_target, cv=10)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_train,train_target, test_size=0.20, random_state=314, stratify=train_target)


# In[ ]:


import lightgbm as lgb

lgb_fit_params={"early_stopping_rounds":50, 
            "eval_metric" : 'binary_logloss', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose':100
           }

lgb_params = {'boosting_type': 'rf',
 'objective': 'binary',
 'metric': 'binary_logloss',
 'verbose': 1,
 'bagging_fraction': 0.8,
 'bagging_freq': 1,
 'num_class': 1,
 'feature_fraction': 0.8,
 'lambda_l1': 0.01,
 'lambda_l2': 0.01,
 'learning_rate': 0.1,
 'max_bin': 255,
 'max_depth': 20,
 'min_data_in_bin': 1,
 'min_data_in_leaf': 1,
 'num_leaves': 31}
lgb_params


# In[ ]:


clf_lgb = lgb.LGBMClassifier(n_estimators=10000, **lgb_params, random_state=123456789, n_jobs=-1)
clf_lgb.fit(X_train, y_train, **lgb_fit_params)
clf_lgb.best_iteration_


# In[ ]:


clf_lgb_fulldata = lgb.LGBMClassifier(n_estimators=int(clf_lgb.best_iteration_), **lgb_params)
clf_lgb_fulldata.fit(new_train, train_target)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestClassifier\nclf_rf_fulldata=RandomForestClassifier(n_estimators=3000, max_features=0.5)\nclf_rf_fulldata.fit(new_train, train_target)')


# In[ ]:


predictions = np.mean((clf_lgb_fulldata.predict_proba(new_test), 
                       clf_rf_fulldata.predict_proba(new_test)), axis=0)
predictions_1 = np.argmax(predictions, axis=1)


# In[ ]:


submission = pd.DataFrame([test['ID'], predictions_1], index=['ID', 'Won_Championship']).T
submission.to_csv('submission-1.csv', index=False)
submission.head()


# LightGBM classifier hyperparameter optimization via scikit-learn's GridSearchCV
# 

# In[ ]:


from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


print('Best parameters found by grid search are:', gridsearch.best_params_)


# LightGBM Hyperparameters + early stopping

# In[ ]:


gbm = lgb.LGBMClassifier(learning_rate = 0.15, metric = 'l1', 
                        n_estimators = 60, boosting_type= 'dart',max_depth =10)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)


# In[ ]:


y_pred = gbm.predict(new_test, num_iteration=gbm.best_iteration_)


# In[ ]:


submission = pd.DataFrame([test['ID'], y_pred], index=['ID', 'Won_Championship']).T
submission.to_csv('submission-2.csv', index=False)
submission.head()


# Feature Importances Graph

# In[ ]:



ax = lgb.plot_importance(gbm, height = 0.8, 
                         max_num_features = 25, 
                         xlim = (0,300), ylim = (0,9), 
                         figsize = (10,10))
plt.show()


# Dimensionality reduction using feature importances
# 

# In[ ]:


# For each feature of our dataset, the result of the following
# code snippet contains numbers of times a feature is used in a model.
sorted(gbm.feature_importances_,reverse=True)


# In[ ]:


# The code below aims to drop  to keep the features that are included in the most important features. 
temp = 0 
total = sum(gbm.feature_importances_)
for feature in sorted(gbm.feature_importances_, reverse=True):
    temp+=feature
    print(feature)
    if temp/total >= 0.85:
        print(feature,temp/total) # stop when we 
        break


# In[ ]:


new_test.columns


# In[ ]:


new_train_1 = new_train.drop(['Coach_Experience_Level','Playing_Style','Team_Value','Previous_SB_Wins'],axis=1)
new_test_1 = new_test.drop(['Coach_Experience_Level','Playing_Style','Team_Value','Previous_SB_Wins'],axis=1)


# In[ ]:


#The above means let go of all variables after PAY_AMT_5
y_pred_prob = gbm.predict_proba(X_test)[:, 1]
auc_roc_0 = str(roc_auc_score(y_test, y_pred_prob)) # store AUC score without dimensionality reduction
print('AUC without dimensionality reduction: \n' + auc_roc_0)


# In[ ]:


# Remake our test/train set with our reduced dataset
X_train, X_test, y_train, y_test = train_test_split(new_train_1, train_target, test_size=0.1, random_state=21)

reduc_estimator = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)

# Parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}

gridsearch = GridSearchCV(reduc_estimator, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set = [(X_test, y_test)],
        eval_metric = ['auc', 'binary_logloss'],
        early_stopping_rounds = 5)
print('Best parameters found by grid search are:', gridsearch.best_params_)


# In[ ]:


print('Best parameters found by grid search are:', gridsearch.best_params_)


# In[ ]:


gbm = lgb.LGBMClassifier(learning_rate = 0.175, metric = 'l1', 
                        n_estimators = 30)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)


# In[ ]:


X_train


# In[ ]:


y_pred = gbm.predict(new_test_1, num_iteration=gbm.best_iteration_)


# In[ ]:


submission = pd.DataFrame([test['ID'], y_pred], index=['ID', 'Won_Championship']).T
submission.to_csv('submission-9.csv', index=False)
submission.head()


# In[ ]:


y_pred_prob = gbm.predict_proba(X_test)[:, 1]


# In[ ]:


y_pred_prob


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for who will win the match')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.grid(True)


# In[ ]:





# In[ ]:




