#!/usr/bin/env python
# coding: utf-8

# # Iris Species Classification Ensemble methods(RandomForest, GradientBoosting, XGBoost)
# **by. YHJ**
# 
# 
# 1.  Data Loding
# 2. Feature Engineering 
# 3. Exploratory Data Analysis (EDA)
# 4. Cross Vaildate - StratifiedKFold
# 5. Hyperparameter Grid setting (We using RandomizedSearchCV)
# 6. Define Model RandomForest/ GradientBoosting/ XGB
# 7. Evaluate Model
# 8. Hyperparameter optimization by Skopt

# In[ ]:


# ###################################################################
# Iris
# ###################################################################

import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Error : /opt/conda/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use array.size > 0 to check that an array is not empty. if diff:
# 
# This problem is nothing. you can ignore this error msg -> refer : https://github.com/scikit-learn/scikit-learn/pull/9816

# In[ ]:


# ###################################################################
# data loding
data = pd.read_csv('../input/Iris.csv')


# In[ ]:


# ###################################################################
# data check
data.head(10)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


species_list = data['Species'].unique()


# In[ ]:


# ###################################################################
# Feature Engineering
# Delet Id, Encoding 'Species'
data = data.drop(['Id'], axis=1)
data['Species'] = LabelEncoder().fit_transform(data['Species'].values)


# In[ ]:


# ###################################################################
# Exploratory Data Analysis (EDA)

# Iris Pairplot
sns.pairplot(data, hue='Species')


# In[ ]:


sns.pairplot(data, x_vars=['SepalLengthCm', 'SepalWidthCm'], y_vars=['PetalLengthCm', 'PetalWidthCm'], hue='Species', size = 4)


# In[ ]:


sns.pairplot(data, x_vars=['SepalLengthCm', 'PetalLengthCm'], y_vars=['PetalWidthCm', 'SepalWidthCm'], hue='Species', size = 4)


# In[ ]:


# Species Distribution
fig = plt.figure(figsize=(12,7))
fig.suptitle('Species Distribution', fontsize=20)

ax1 = fig.add_subplot(221)
data.groupby(['Species']).PetalLengthCm.plot('hist', alpha=0.8, title='PetalLengthCm')
plt.legend(species_list, loc=1, fontsize='10')
ax2 = fig.add_subplot(222,sharey=ax1)
data.groupby(['Species']).PetalWidthCm.plot('hist', alpha=0.8, title='PetalWidthCm')
plt.legend(species_list, loc=1, fontsize='10')
ax3 = fig.add_subplot(223,sharey=ax1)
data.groupby(['Species']).SepalLengthCm.plot('hist', alpha=0.8, title='SepalLengthCm')
plt.legend(species_list, loc=1, fontsize='10')
ax4 = fig.add_subplot(224,sharey=ax1)
data.groupby(['Species']).SepalWidthCm.plot('hist', alpha=0.8, title='SepalWidthCm')
plt.legend(species_list, loc=1, fontsize='10')

plt.show()


# In[ ]:


# ###################################################################
# Explanatory variable X Response variable Y
X = data.drop(['Species'], axis=1)
y = data['Species']


# In[ ]:


# ###################################################################
# Data Split Ver.1
seed = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)


# In[ ]:


# ###################################################################
# Cross Vaildate - KFold Ver.1
# Regression -> KFold
# classifier -> StratifiedKFold

# Kfold = KFold(n_splits=5, random_state=seed)
Kfold = StratifiedKFold(n_splits=5, random_state=seed)


# In[ ]:


# ###################################################################
# Visualization & Report

# GridSearchCV, RandomizedSearchCV Report Function 
# -> by. scikit-learn.org "Comparing randomized search and grid search for hyperparameter estimation"
def report(results, n_top=3):
    lcount = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            if lcount > 2:
                break
            lcount += 1
                
def model_scores(y_test,y_pred):
    acc = accuracy_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print('Accuracy : %.2f %%' % acc)
    print('F1 Score : %.2f ' % f1)
    print('Confusion Matrix :')
    print(conf_matrix)
    
    # sns.heatmap(conf_matrix, cmap='Greys', annot=True, linewidths=0.5)

    #print(classification_report(y_test, y_pred))
    # Warning : UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
    # when no data points are classified as positive, precision divides by zero as it is defined as TP / (TP + FP) (i.e., true positives / true and false positives). 
    # The library then sets precision to 0, but issues a warning as actually the value is undefined. 
    # F1 depends on precision and hence is not defined either.
    
    return {'acc':[acc], 'f1':[f1]}


def result_vis(acc_results_vis, f1_results_vis, names_vis):   
    fig =plt.figure(figsize=(5,5))
    fig.suptitle('Algorithm Comparison - Accuracy')
    ax = fig.add_subplot(111)
    sns.barplot(data=acc_results_vis)
    ax.set_xticklabels(names_vis)
    plt.show()

    fig =plt.figure(figsize=(5,5))
    fig.suptitle('Algorithm Comparison - F1')
    ax = fig.add_subplot(111)
    sns.barplot(data=f1_results_vis)
    ax.set_xticklabels(names_vis)
    plt.show()


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# In[ ]:


# ###################################################################
# Hyperparameter Grid Ver.1

param_grid_pipe = {
        'RandomForest' : {'RandomForest__n_estimators'      : st.randint(500, 1000),
                          'RandomForest__max_features'      : ['auto','sqrt','log2'],
                          'RandomForest__max_depth'         : st.randint(2, 100),
                          'RandomForest__min_samples_split' : st.randint(2, 100),
                          'RandomForest__min_samples_leaf'  : st.randint(2, 100),
                          'RandomForest__criterion'         : ['gini', 'entropy']},
                          
        'GBM'          : {'GBM__n_estimators'               : st.randint(500, 1000),
                          'GBM__max_depth'                  : st.randint(2, 100),
                          'GBM__learning_rate'              : st.uniform(0.001, 0.2),
                          'GBM__min_samples_split'          : st.randint(2, 100),
                          'GBM__min_samples_leaf'           : st.randint(2, 100)},
                          
        'XGB'          : {'XGB__n_estimators'               : st.randint(500, 1000),
                          'XGB__max_depth'                  : st.randint(2, 100),
                          'XGB__learning_rate'              : st.uniform(0.001, 0.2),
                          'XGB__colsample_bytree'           : st.beta(10, 1),
                          'XGB__subsample'                  : st.beta(10, 1),
                          'XGB__gamma'                      : st.uniform(0, 10),
                          'XGB__min_child_weight'           : st.expon(0, 10)}
}


# In[ ]:


# Using Pipeline Casing
cache_dir = mkdtemp()


# In[ ]:


# ###################################################################
# MODELS - Pipeline
# StandardScaler + Model
models = []
models.append(('RandomForest _pipe', RandomizedSearchCV(Pipeline([('scaler', StandardScaler()), ('RandomForest', RandomForestClassifier())], memory=cache_dir), param_grid_pipe['RandomForest'], scoring='accuracy', cv=Kfold, n_jobs=-1, n_iter=100)))
models.append(('GBM          _pipe', RandomizedSearchCV(Pipeline([('scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())], memory=cache_dir), param_grid_pipe['GBM'], scoring='accuracy', cv=Kfold, n_jobs=-1, n_iter=100)))
models.append(('XGBoost      _pipe', RandomizedSearchCV(Pipeline([('scaler', StandardScaler()), ('XGB', xgb.XGBClassifier(booster='gbtree',objective='multi:softmax'))], memory=cache_dir), param_grid_pipe['XGB'], scoring='accuracy', cv=Kfold, n_jobs=-1, n_iter=100)))


# In[ ]:


# ###################################################################
# MODELS Scores Ver.1
# Evaluate each model

acc_results =[]
f1_results =[]
names= []
for name, model in models:
    
    start = time()
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)
    
    print('')
    print('## %s ##################################' % name)
    print('Best  score : %.4f' % model.best_score_)
    print('Test  score : %.4f' % model.score(X_test, y_test))
    results = model_scores(y_test, y_pred)
    
    print("\n%s ParamsSearchCV took %.2f seconds for %d candidate parameter settings." 
          % (name.replace(" ", ""), time() - start, len(model.cv_results_['params'])))
    report(model.cv_results_)
    
    acc_results.append(results['acc'])
    f1_results.append(results['f1'])
    names.append(name)
    rmtree(cache_dir)


# **1. RandomForest** (407.4 s)
# * Test Accuracy : 84.21 %
# * CV(5) Best Accuracy : 73.68 %
# 
# **2. GradientBoosting ** (311.01 s)
# * Test Accuracy : 100 %% !!!
# * CV(5) Best Accuracy : 94.74 %
# 
# **3. XGBoost** (29.71 s)
# * Test Accuracy : 97.37 %
# * CV(5) Best Accuracy : 92.11 %

# In[ ]:


result_vis(acc_results, f1_results, names)


# **Model Compare chart**
# * Best scoring model is GBM, and 2nd is XGB
# * However, the result may vary depending on the Hyperparameter setting.
# * therefore We tring Hyperparameter optimization
# * I used Skopt (Other things HyperOpt(TPE), spearmint(Bayesian optimization))

# In[ ]:


# ###################################################################
# Hyperparameter optimization by Skopt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# In[ ]:


# ###################################################################
# Hyperparameter Grid_Skopt Ver.1
# Categorical
# Real
# Integer

param_grid_skopt = {
        'RandomForest' : {'RandomForest__n_estimators'      : Integer(1000, 10000),
                          'RandomForest__max_features'      : Categorical(['auto','sqrt','log2']),
                          'RandomForest__max_depth'         : Integer(2, 500),
                          'RandomForest__min_samples_split' : Integer(2, 500),
                          'RandomForest__min_samples_leaf'  : Integer(2, 500),
                          'RandomForest__criterion'         : Categorical(['gini', 'entropy'])},

        'GBM'          : {'GBM__n_estimators'               : Integer(1000, 10000),
                          'GBM__max_depth'                  : Integer(2, 500),
                          'GBM__learning_rate'              : Real(1e-6, 1e-1, 'log-uniform'),
                          'GBM__min_samples_split'          : Integer(2, 500),
                          'GBM__min_samples_leaf'           : Integer(2, 500)},
                          
        'XGB'          : {'XGB__booster'                    : Categorical(['gbtree','dart']),
                          'XGB__objective'                  : Categorical(['multi:softmax','multi:softprob']),
                          'XGB__n_estimators'               : Integer(1000, 10000),
                          'XGB__max_depth'                  : Integer(2, 500),
                          'XGB__learning_rate'              : Real(1e-6, 1e-1, 'log-uniform'),
                          'XGB__subsample'                  : Real(0, 1, 'uniform'),
                          'XGB__gamma'                      : Integer(0, 100),
                          'XGB__min_child_weight'           : Real(1e-2, 1e+2, 'log-uniform')}
}


# In[ ]:


cache_dir = mkdtemp()


# In[ ]:


# ###################################################################
# MODELS - Skopt (BayesSearchCV)
# Bayesian optimization over hyper parameters
# ###################################################################
models = []
models.append(('RandomForest', BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('RandomForest', RandomForestClassifier())], memory=cache_dir), param_grid_skopt['RandomForest'], n_iter = 50, n_jobs=-1, cv=Kfold)))
models.append(('GBM         ', BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())], memory=cache_dir), param_grid_skopt['GBM'], n_iter = 50, n_jobs=-1, cv=Kfold)))
models.append(('XGBoost     ', BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('XGB', xgb.XGBClassifier())], memory=cache_dir), param_grid_skopt['XGB'], n_iter = 50, n_jobs=-1, cv=Kfold)))


# In[ ]:


# ###################################################################
# MODELS Scores Ver.1
# Evaluate each model

acc_results =[]
f1_results =[]
names= []
for name, model in models:
    
    start = time()
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)
    
    print('')
    print('## %s ##################################' % name)
    print('Best  score : %.4f' % model.best_score_)
    print('Test  score : %.4f' % model.score(X_test, y_test))
    results = model_scores(y_test, y_pred)
    
    print("\n%s BayesSearchCV took %.2f seconds for %d candidate parameter settings." 
          % (name.replace(" ", ""), time() - start, len(model.cv_results_['params'])))
    print("Best Parameters : ", model.best_params_)
    #report(model.cv_results_)
    
    acc_results.append(results['acc'])
    f1_results.append(results['f1'])
    names.append(name)
    rmtree(cache_dir)


# In[ ]:


result_vis(acc_results, f1_results, names)

