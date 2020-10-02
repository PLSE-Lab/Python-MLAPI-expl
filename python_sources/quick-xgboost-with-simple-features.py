#!/usr/bin/env python
# coding: utf-8

# Predicting the presence of mosquito species.
# --------------------------------------------
# 
# Hi, friends.
# 
# Let's make a first attempt at predicting whether a location is inhabited by *Aedes aegypti* or *Aedes albopictus*. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import seaborn as sns 

data = pd.read_csv('../input/aegypti_albopictus.csv')
print(data.head())


# Let's examine the five most common values for each variable, as well as the number of unique values for each.

# In[ ]:


for c in data.columns:
    counts = data[c].value_counts()
    print("\nFrequency of each value of %s:\n" % c)
    print(counts.head())
    print("\nUnique values of %s: %i\n" % (c, len(counts)))


# It appears that (Polygon_Admin = -999) has a one-to-one correspondence to (Location_Type = point), so perhaps there's no urgency in imputing or removing this value when encoding categories. Similarly, the cardinality of GAUL_AD0 seems to match that of Country and Country_ID, so we'll assume that the latter two can be dropped for now. This assumption is reasonable considering that the GAUL (global administrative unit layer code) is unique to each country as described in the reference.
# 
# For a quick visual, we'll encode categories with integers and plot the correlations.

# In[ ]:


data = data.drop(['OCCURRENCE_ID', 'COUNTRY', 'COUNTRY_ID'], axis=1)

categoricals = ['VECTOR', 'SOURCE_TYPE', 'LOCATION_TYPE', 
                'POLYGON_ADMIN', 'GAUL_AD0', 'YEAR', 'STATUS']

for c in categoricals:
    data[c] = pd.factorize(np.array(data[c]))[0]

print(data.head())

sns.heatmap(data.corr())

plt.show()


# Note that Vector is the target variable, which indicates whether the species is *Aedes aegypti* or *Aedes albopictus*, encoded as 1 and 0, respectively. It doesn't seem to correlate strongly with any other variable.
# 
# I'll update this notebook with more plots in the future when I attempt to extract new features. For now, we can already see below that some distinct clusters occur for each class in the scatterplots for Location_Type and Polygon_Admin. Maybe some informative features can be built from these clues.

# In[ ]:


grid = sns.FacetGrid(data, hue='VECTOR', row='LOCATION_TYPE')
grid.map(plt.scatter, 'X', 'Y')
grid.add_legend()

plt.show()


# In[ ]:


grid = sns.FacetGrid(data, hue='VECTOR', row='POLYGON_ADMIN')
grid.map(plt.scatter, 'X', 'Y')
grid.add_legend()

plt.show()


# First, we'll cross-validate with XGBoost to see the best testing scores after about a hundred rounds. Then, we'll evaluate predictions using several metrics after fitting a random 80/20 train/test split.

# In[ ]:


import xgboost as xgb 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score

X = data.ix[:, 1:]
y = data.ix[:, 0]

# Instantiate XGBoost
n_estimators = 150
dtrain = xgb.DMatrix(X, y)

# XGBoost was tuned on the raw data.
bst = XGBClassifier(n_estimators=n_estimators,
                    max_depth=3, 
                    min_child_weight=5, 
                    gamma=0.5, 
                    learning_rate=0.05, 
                    subsample=0.7, 
                    colsample_bytree=0.7, 
                    reg_alpha=0.001,
                    seed=1,
                    silent=True)

# Cross-validate XGBoost
params = bst.get_xgb_params() # Extract parameters from XGB instance to be used for CV
num_boost_round = bst.get_params()['n_estimators'] # XGB-CV has different names than sklearn

cvresult = xgb.cv(params, dtrain, num_boost_round=num_boost_round, 
                  nfold=10, metrics=['logloss', 'auc', 'error'], seed=1)

# XGBoost summary
print("="*80)
print("\nXGBoost summary for 150 rounds of 10-fold cross-validation:")
print("\nBest mean log-loss: %.4f" % cvresult['test-logloss-mean'].min())
print("\nBest mean AUC: %.4f" % cvresult['test-auc-mean'].max())
print("\nBest mean error: %.4f" % cvresult['test-error-mean'].min())
print("="*80)

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

bst.fit(X_train, y_train, eval_metric='logloss')
pred = bst.predict(X_test)

print("="*80)
print("\nXGBoost performance on unseen data:")
print("\nlog-loss: %.4f" % log_loss(y_test, pred))
print("\nAUC: %.4f" % roc_auc_score(y_test, pred))
print("\nF1 score: %.4f" % f1_score(y_test, pred))
print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))
print("="*80)


# XGBoost seems to do quite well in the best averages of log-loss, AUC, and error, evaluated across rounds of boosting. (Averages are evaluated across folds of cross-validation.) It's possible that I made a mistake. Please let me know by leaving a comment below.
# 
# For amusement, let's also try implementing some custom features based on clusters found in the scatterplots. Perhaps these can improve performance.

# In[ ]:


# Rename X temporarily to avoid confusion with the X variable
data = X

# -100 < X < -70     
# 20 < Y < 40                           
X_neg100_neg70 = np.where(data['X'] > -100, 1, 0) - np.where(data['X'] > -70, 1, 0)     
                                           
Y_20_40 = np.where(data['Y'] > 20, 1, 0) - np.where(data['Y'] > 40, 1, 0)     
                                      
data['X_neg100_neg70_Y_20_40'] = X_neg100_neg70*Y_20_40     
                                        
# -110 < X < -50               
# -40 < Y < 30                      
X_neg110_neg50 = np.where(data['X'] > -110, 1, 0) - np.where(data['X'] > -50, 1, 0)     
                                                                                          
Y_neg40_30 = np.where(data['Y'] > -40, 1, 0) - np.where(data['Y'] > 30, 1, 0)     
                                                                      
data['X_neg110_neg50_Y_neg40_30'] = X_neg110_neg50*Y_neg40_30

X = data
print(X.head())


# In[ ]:


dtrain = xgb.DMatrix(X, y)

cvresult = xgb.cv(params, dtrain, num_boost_round=num_boost_round, 
                  nfold=10, metrics=['logloss', 'auc', 'error'], seed=1)

# XGBoost summary
print("="*80)
print("\nXGBoost summary for 150 rounds of 10-fold cross-validation:")
print("\nBest mean log-loss: %.4f" % cvresult['test-logloss-mean'].min())
print("\nBest mean AUC: %.4f" % cvresult['test-auc-mean'].max())
print("\nBest mean error: %.4f" % cvresult['test-error-mean'].min())
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

bst.fit(X_train, y_train, eval_metric='logloss')
pred = bst.predict(X_test)

print("="*80)
print("\nXGBoost performance on unseen data:")
print("\nlog-loss: %.4f" % log_loss(y_test, pred))
print("\nAUC: %.4f" % roc_auc_score(y_test, pred))
print("\nF1 score: %.4f" % f1_score(y_test, pred))
print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))
print("="*80)


# Only slight improvements were made in all metrics. Anyway, it was worth a try.
# 
# I'll return to this notebook another day to attempt more plots, features, and models. Until then, I hope you find something useful here. 
# 
# Best wishes.
# 
# *The rest of this project can be found on my GitHub:* <br>
# [https://github.com/Justin-Le/predicting-zika-virus][1]
# 
#   [1]: https://github.com/Justin-Le/predicting-zika-virus
