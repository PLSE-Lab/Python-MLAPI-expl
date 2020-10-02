#!/usr/bin/env python
# coding: utf-8

# ## Acknowledgements
# #### This kernel uses such good kernels
#    - https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
#    - https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
#    - https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
#    - https://www.kaggle.com/startupsci/titanic-data-science-solutions
#    - https://www.kaggle.com/kabure/titanic-eda-model-pipeline-keras-nn
#    - https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
#    - https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg\
#    - https://www.kaggle.com/littleraj30/eda-model-building-in-depth-on-heart-disease
#    - https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [EDA](#3)
# 1. [FE: building the feature importance diagrams](#4)
#     -  [4.1 LGBM](#4.1)
#     -  [4.2 XGB](#4.2) 
#     -  [4.3 Logistic Regression](#4.3) 
#     -  [4.4 Linear Regression](#4.4)
# 1. [Comparison of the all feature importance diagrams ](#5)
# 1. [Preparing to modeling](#6)  
# 1. [Tuning models and test for all features](#7)
#     -  [Decition Tree Classifiter](#7.1)
#     -  [XGB Classifier](#7.2)
#     -  [Support Vector Machines](#7.3) 
# 1. [Models evaluation](#8)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas_profiling as pp

# models
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# NN models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression,LogisticRegression, SGDRegressor, RidgeCV

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


data = pd.read_csv("../input/heart-disease-uci/heart.csv")


# In[ ]:


data = data[data['age'] < 80]


# In[ ]:


data.head(3)


# In[ ]:


data.info()


# ## 3. EDA <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


pp.ProfileReport(data)


# In[ ]:


# Thanks to: https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients
plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# In[ ]:


#Thanks to: https://www.kaggle.com/littleraj30/eda-model-building-in-depth-on-heart-disease
fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='fbs',data=data,hue='target',palette='Set3',ax=ax[0])
ax[0].set_xlabel("0-> fps <120 , 1-> fps>120",size=12)
data.fbs.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0],cmap='Blues')
ax[1].set_title("0 -> fps <120 , 1 -> fps>120",size=12)


# In[ ]:


#Thanks to: https://www.kaggle.com/littleraj30/eda-model-building-in-depth-on-heart-disease
fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='restecg',data=data,hue='target',palette='Set3',ax=ax[0])
ax[0].set_xlabel("resting electrocardiographic",size=12)
data.restecg.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,
                                     explode=[0.005,0.05,0.05],cmap='Oranges')
ax[1].set_title("resting electrocardiographic",size=12)


# In[ ]:


#Thanks to: https://www.kaggle.com/littleraj30/eda-model-building-in-depth-on-heart-disease
fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='slope',data=data,hue='target',palette='Set1',ax=ax[0])
ax[0].set_xlabel("peak exercise ST segment",size=12)
data.slope.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,explode=[0.005,0.05,0.05],cmap='Reds')

ax[1].set_title("peak exercise ST segment ",size=12)


# In[ ]:


#Thanks to: https://www.kaggle.com/littleraj30/eda-model-building-in-depth-on-heart-disease
fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='ca',data=data,hue='target',palette='Set2',ax=ax[0])
ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)
data.ca.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Oranges')
ax[1].set_title("number of major vessels colored by flourosopy",size=12)


# In[ ]:


#Thanks to: https://www.kaggle.com/littleraj30/eda-model-building-in-depth-on-heart-disease
fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='thal',data=data,hue='target',palette='Set2',ax=ax[0])
ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)
data.thal.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greens')
ax[1].set_title("number of major vessels colored by flourosopy",size=12)


# In[ ]:


# Thanks to: https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients
male =len(data[data['sex'] == 1])
female = len(data[data['sex']== 0])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Male','Female'
sizes = [male,female]
colors = ['pink', 'darkgreen']
explode = (0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients
plt.figure(figsize=(8,6))

# Data to plot
labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'
sizes = [len(data[data['cp'] == 0]),len(data[data['cp'] == 1]),
         len(data[data['cp'] == 2]),
         len(data[data['cp'] == 3])]
colors = ['pink', 'yellowgreen','purple','gold']
explode = (0, 0,0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients
plt.figure(figsize=(8,6))

# Data to plot
labels = 'fasting blood sugar < 120 mg/dl','fasting blood sugar > 120 mg/dl'
sizes = [len(data[data['fbs'] == 0]),len(data[data['cp'] == 1])]
colors = ['grey', 'yellowgreen','orange','gold']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients
plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'target',palette='GnBu')
plt.show()


#  ## 4. FE: building the feature importance diagrams <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# <a class="anchor" id="4.1"></a>
# ### 4.1 LGBM 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Clone data for FE 
train_fe = copy.deepcopy(data)
target_fe = train_fe['target']
del train_fe['target']


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
X = train_fe
z = target_fe


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }

modelL = lgb.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgb.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
feature_score = pd.DataFrame(train_fe.columns, columns = ['feature']) 
feature_score['score_lgb'] = modelL.feature_importance()


# <a class="anchor" id="4.2"></a>
# ### 4.2 XGB
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:squarederror',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=10)

print('score = %1.5f, n_boost_round =%d.'%(modelx.best_score,modelx.best_iteration))


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
feature_score


# <a class="anchor" id="4.3"></a>
# ### 4.3 Logistic Regression
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Standardization for regression model
train_fe = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(train_fe),
    columns=train_fe.columns,
    index=train_fe.index
)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train_fe, target_fe)
coeff_logreg = pd.DataFrame(train_fe.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# the level of importance of features is not associated with the sign
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()
feature_score = pd.merge(feature_score, coeff_logreg, on='feature')


# <a class="anchor" id="4.4"></a>
# ### 4.4 Linear Regression
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Linear Regression

linreg = LinearRegression()
linreg.fit(train_fe, target_fe)
coeff_linreg = pd.DataFrame(train_fe.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')
feature_score


# <a class="anchor" id="5"></a>
# ## 5. Comparison of the all feature importance diagrams 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Thanks to: https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
# MinMax scale all importances
feature_score = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(feature_score),
    columns=feature_score.columns,
    index=feature_score.index
)

# Create mean column
feature_score['mean'] = feature_score.mean(axis=1)

# Plot the feature importances
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('mean', ascending=False)


# In[ ]:


# Thanks to: Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Create total column with different weights
feature_score['total'] = 0.5*feature_score['score_lgb'] + 0.3*feature_score['score_xgb']                        + 0.1*feature_score['score_logreg'] + 0.1*feature_score['score_linreg']

# Plot the feature importances
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('total', ascending=False)


#  ## 6. Preparing to modeling <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


target_name = 'target'
data_target = data[target_name]
data = data.drop([target_name], axis=1)


# In[ ]:


train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.2, random_state=0)


# ## 7. Tuning models and test for all features <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 7.1 Decition Tree Classifiter <a class="anchor" id="7.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Decision Tree Classifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, target)
acc_decision_tree = round(decision_tree.score(train, target) * 100, 2)
acc_decision_tree


# In[ ]:


acc_test_decision_tree = round(decision_tree.score(test, target_test) * 100, 2)
acc_test_decision_tree


# ### 7.2 XGB Classifier <a class="anchor" id="7.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def hyperopt_xgb_score(params):
    clf = XGBClassifier(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_xgb = {
            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.005),
            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.005),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.005),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': 1,
            'missing': None
        }
 
best = fmin(fn=hyperopt_xgb_score, space=space_xgb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[ ]:


params = space_eval(space_xgb, best)
params


# In[ ]:


XGB_Classifier = XGBClassifier(**params)
XGB_Classifier.fit(train, target)
acc_XGB_Classifier = round(XGB_Classifier.score(train, target) * 100, 2)
acc_XGB_Classifier


# In[ ]:


acc_test_XGB_Classifier = round(XGB_Classifier.score(test, target_test) * 100, 2)
acc_test_XGB_Classifier


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(XGB_Classifier,ax = axes,height =0.5)
plt.show();
plt.close()


# ### 7.3 Support Vector Machines 
# <a class="anchor" id="7.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


svc = SVC()
svc.fit(train, target)
acc_svc = round(svc.score(train, target) * 100, 2)
acc_svc


# In[ ]:


acc_test_svc = round(svc.score(test, target_test) * 100, 2)
acc_test_svc


# ## 8. Models evaluation <a class="anchor" id="8"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


models = pd.DataFrame({
    'Model': ['Decision Tree Classifier','XGBClassifier', 'Support Vector Machines'],
    
    'Score_train': [acc_decision_tree, acc_XGB_Classifier, acc_svc],
    'Score_test': [acc_test_decision_tree, acc_test_XGB_Classifier, acc_test_svc]
                    })


# In[ ]:


models.sort_values(by=['Score_train', 'Score_test'], ascending=False)


# In[ ]:


models.sort_values(by=['Score_test', 'Score_train'], ascending=False)


# In[ ]:


models['Score_diff'] = abs(models['Score_train'] - models['Score_test'])
models.sort_values(by=['Score_diff'], ascending=True)


# In[ ]:


# Plot
plt.figure(figsize=[25,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['Score_train'], label = 'Score_train')
plt.plot(xx, models['Score_test'], label = 'Score_test')
plt.legend()
plt.title('Score of 20 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('Score, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()

