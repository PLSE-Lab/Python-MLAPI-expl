#!/usr/bin/env python
# coding: utf-8

# Initial version forked from SCTP-working(LGB) https://www.kaggle.com/dromosys/sctp-working-lgb courtesy of Dromosys

# # How can we genetically engineer features?
# 
# ## Executive Summary
# We would like to find a mathematical formula that will create a new feature from the ones we have - and therefore give the machine learning algorithms more to work with. In the case of the Santander competition, this is dificult because the features are already well pre-processed. In other words, they are already statistically independent and seem to contain little, if any, redundant information (as if principal component analysis has already been performed). Instead, we can turn to generically engineered formulas to create new features. 
# 
# 
# Specificaly, if we use 199 of the 200 features to create an estimation of the missing feature - this will be a imperfect estimation, because as noted above, the features do not contain redundant information. Nevertheless, this poor estimation can be considered to be a "new feature" or a new classification of the current 199 features as being a member of the missing feature target. This can be repeated for every feature, giving us 200 new features to add to the training (and testing) sets.
# 
# This kernel demonstrates this, and uses "gplearn" which extends the scikit-learn machine learning library to perform Genetic Programming (GP) with symbolic regression.
# 
# ## Background
# From the website https://gplearn.readthedocs.io/en/stable/intro.html ...
# 
# "Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression that best describes a relationship. It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets in order to predict new data. Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations."
# 

# In[ ]:


# INPORTING WHAT WE NEED
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import os
import time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV

# GENETIC ALGORITHM
import gplearn
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
print('Rows: ',train_df.shape[0],'Columns: ',train_df.shape[1])
train_df.info()


# In[ ]:


train_df.head()


# In[ ]:


train_df['target'].value_counts()


# In[ ]:


sns.countplot(train_df['target'])
sns.set_style('whitegrid')


# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


X_test1 = test_df.drop('ID_code',axis=1)


# In[ ]:


X1 = train_df.drop(['ID_code','target'],axis=1)


# In[ ]:


# Create a fitness function that is the mean absolute percentage error
def _my_fit(y, y_pred, w):
    diffs = np.abs(y - y_pred)  
    return 100. * np.average(diffs, weights=w)
my_fit = make_fitness(_my_fit, greater_is_better=False)


# In[ ]:


# Choose the mathematical functions we will combine together
function_set = ['add', 'sub', 'mul', 'div', 'log', 
                'sqrt', 'log', 'abs', 'neg', 'inv', 
                'max', 'min', 
                'sin', 'cos', 'tan' ] 

# Create the genetic learning regressor
gp = SymbolicRegressor(function_set=function_set, metric = my_fit,
                       verbose=1, generations = 3, 
                       random_state=0, n_jobs=-1)


# In[ ]:


# Using NUMPY structures, remove one feature (column of data) at a time from the training set
# Use that removed column as the target for the algorithm
# Use the genetically engineered formula to create the new feature
# Do this for both the training set and the test set

X1a = np.array(X1)
sam = X1a.shape[0]
col = X1a.shape[1]
X2a = np.zeros((sam, col))

X_test1a = np.array(X_test1)
sam_test = X_test1a.shape[0]
col_test = X_test1a.shape[1]
X_test2a = np.zeros((sam_test, col_test))

for i in range(col) :
    X = np.delete(X1a,i,1)
    y = X1a[:,i]
    gp.fit(X, y) 
    X2a[:,i] = gp.predict(X)
    X = np.delete(X_test1a, i, 1)
    X_test2a[:,i] = gp.predict(X)
    
X2 = pd.DataFrame(X2a)
X_test2 = pd.DataFrame(X_test2a) 


# In[ ]:


# Add the new features to the existing 200 features
X = pd.concat([X1, X2], axis=1, sort=False) 
X_test = pd.concat([X_test1, X_test2], axis=1, sort=False)  
y = train_df['target']


# In[ ]:


# See my earlier kernel "Santander-Statistics" https://www.kaggle.com/pnussbaum/santander-statistics-v04
# for reasons why I have moved from "StratifiedKFold" to simply "KFold"

n_fold = 5
# folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)


# In[ ]:


# Used parameters from https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82515 courtesy Nanashi
params = {
    'num_leaves': 13,
    'min_data_in_leaf': 80,
    'objective': 'binary',
    'max_depth': -1,
    'learning_rate': 0.01,
#    'boost': 'gbdt',
    'boosting': 'gbdt',
    'bagging_freq': 5,
    'bagging_fraction': 0.33,
    'feature_fraction': 0.05,
    'metric':'auc',
    'verbosity': 1, 
    'min_sum_hessian_in_leaf': 10.0,
    'num_threads': 12,
    'tree_learner': 'serial',
    'boost_from_average':'false'
}
# params = {'num_leaves': 9,
#          'min_data_in_leaf': 42,
#          'objective': 'binary',
#          'max_depth': 16,
#          'learning_rate': 0.0123,
#          'boosting': 'gbdt',
#          'bagging_freq': 5,
#          'bagging_fraction': 0.8,
#          'feature_fraction': 0.8201,
#          'metric': 'auc',
#          'verbosity': -1,
#          'subsample': 0.81,
#          'min_gain_to_split': 0.01077313523861969,
#          'min_child_weight': 19.428902804238373,
#          'bagging_seed': 11,
#          'reg_alpha': 1.728910519108444,
#          'reg_lambda': 4.9847051755586085,
#          'random_state': 42,
#          'num_threads': 4}


# In[ ]:


prediction = np.zeros(len(X_test))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    model = lgb.train(params,train_data,num_boost_round=20000,
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)
             
    #y_pred_valid = model.predict(X_valid) 
    prediction += model.predict(X_test, num_iteration=model.best_iteration)/n_fold 


# In[ ]:


from catboost import CatBoostClassifier,Pool
train_pool = Pool(X,y) 
m = CatBoostClassifier(iterations=300,eval_metric="AUC", boosting_type = 'Ordered')
m.fit(X,y,silent=True)
y_pred1 = m.predict(X_test)
m.best_score_


# In[ ]:


prediction


# In[ ]:


y_pred1


# In[ ]:


sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = prediction
sub.to_csv("submission.csv", index=False)


# In[ ]:


sub1 = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub1["target"] = y_pred1
sub1.to_csv("submission1.csv", index=False)


# In[ ]:


sub2 = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub2["target"] = (prediction + y_pred1)/2
sub2.to_csv("submission2.csv", index=False)


# ## Reference
# 1. https://www.kaggle.com/gpreda/santander-eda-and-prediction
# 2. https://www.kaggle.com/deepak525/sctp-lightgbm-lb-0-899

# In[ ]:




