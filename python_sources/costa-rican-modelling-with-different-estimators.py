#!/usr/bin/env python
# coding: utf-8

# # Costa Rican Household Poverty Level Prediction
# 
# Problem and Data Explanation
# The data for this competition is provided in two files: train.csv and test.csv. The training set has 9557 rows and 143 columns while the testing set has 23856 rows and 142 columns. Each row represents one individual and each column is a feature, either unique to the individual, or for the household of the individual. The training set has one additional column, Target, which represents the poverty level on a 1-4 scale and is the label for the competition. A value of 1 is the most extreme poverty.
# 
# This is a supervised multi-class classification machine learning problem:
# 
# Supervised: provided with the labels for the training data
# Multi-class classification: Labels are discrete values with 4 classes
# The Target values represent poverty levels as follows:
# 
# 1 = extreme poverty 
# 2 = moderate poverty 
# 3 = vulnerable households 
# 4 = non vulnerable households
# 
# Objectives:
# Objective of this kernel is to perform modeling with the following estimators with default parameters & get accuracy
#         
#         GradientBoostingClassifier
#         RandomForestClassifier
#         KNeighborsClassifier
#         ExtraTreesClassifier
#         XGBoost
#         LightGBM
#         
#         Then perform tuning using Bayesian Optimization & compare the accuracy of the estimators. 
#         In this kerenal very simple code is used so that beginners can understand the code.
#         Core Data fields
# 
# Id - a unique identifier for each row.
# Target - the target is an ordinal variable indicating groups of income levels. 
# 1 = extreme poverty 
# 2 = moderate poverty 
# 3 = vulnerable households 
# 4 = non vulnerable households
# idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
# parentesco1 - indicates if this person is the head of the household.
# This data contains 142 total columns.
# All Data fields
# 
# 

# ## Calling required libraries for the work

# In[ ]:


# essential libraries
import numpy as np 
import pandas as pd
# for data visulization
import matplotlib.pyplot as plt
import seaborn as sns


#for data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# for modeling estimators
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

# for measuring performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#for tuning parameters
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance

# Misc.
import os
import time
import gc
import random
from scipy.stats import uniform
import warnings


# ## Reading the data

# In[ ]:


pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


ids=test['Id']


# ## Explore data and perform data visualization

# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.info()   


# In[ ]:


test


# In[ ]:


sns.countplot("Target", data=train)


# In[ ]:


sns.countplot(x="r4t3",hue="Target",data=train)


# In[ ]:


sns.countplot(x="v18q",hue="Target",data=train)


# In[ ]:


sns.countplot(x="v18q1",hue="Target",data=train)


# In[ ]:


sns.countplot(x="tamhog",hue="Target",data=train)


# In[ ]:


sns.countplot(x="hhsize",hue="Target",data=train)


# In[ ]:


sns.countplot(x="abastaguano",hue="Target",data=train)


# In[ ]:


sns.countplot(x="noelec",hue="Target",data=train)


# In[ ]:


train.select_dtypes('object').head()


# In[ ]:




yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
    
    


# In[ ]:


yes_no_map = {'no':0,'yes':1}
test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)
test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)
test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)


# ## Converting categorical objects into numericals 

# In[ ]:


train[["dependency","edjefe","edjefa"]].describe()


# ### Fill in missing values (NULL values)  using 1 for yes and 0 for no

# In[ ]:


# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[ ]:


train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)


# In[ ]:


train['v2a1'] = train['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)


# In[ ]:


train['rez_esc'] = train['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)


# In[ ]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[ ]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# ### Dropping unnecesary columns

# In[ ]:


train.drop(['Id','idhogar'], inplace = True, axis =1)

test.drop(['Id','idhogar'], inplace = True, axis =1)


# In[ ]:


train.shape


# In[ ]:


test.shape


# ### Dividing the data into predictors & target

# In[ ]:


y = train.iloc[:,140]
y.unique()


# In[ ]:


X = train.iloc[:,1:141]
X.shape


# ### Scaling  numeric features & applying PCA to reduce features

# In[ ]:


my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)
#pca = PCA(0.95)
#X = pca.fit_transform(X)


# In[ ]:


X.shape


# In[ ]:


#subjecting the same to test data
my_imputer = SimpleImputer()
test = my_imputer.fit_transform(test)
scale = ss()
test = scale.fit_transform(test)
#pca = PCA(0.95)
#test = pca.fit_transform(test)


# ### Final features selected for modeling

# In[ ]:


X.shape, y.shape,test.shape


# ### Splitting the data into train & test 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# # Modelling

# ## Modelling with Random Forest

# In[ ]:



modelrf = rf()


# In[ ]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


classes = modelrf.predict(X_test)


# In[ ]:


(classes == y_test).sum()/y_test.size 


# In[ ]:


f1 = f1_score(y_test, classes, average='macro')
f1


# ## Performing tuning using Bayesian Optimization.

# In[ ]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    rf(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[ ]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[ ]:


modelrfTuned=rf(criterion="entropy",
               max_depth=77,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[ ]:


start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


yrf=modelrfTuned.predict(X_test)


# In[ ]:


yrf


# In[ ]:


yrftest=modelrfTuned.predict(test)


# In[ ]:


yrftest


# In[ ]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[ ]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# ### Accuracy improved from 71.91% to 76.20%

# ## Modelling with ExtraTreeClassifier

# In[ ]:


modeletf = ExtraTreesClassifier()


# In[ ]:


start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


classes = modeletf.predict(X_test)

classes


# In[ ]:


(classes == y_test).sum()/y_test.size


# In[ ]:


f1 = f1_score(y_test, classes, average='macro')
f1


# ## Performing tuning using Bayesian Optimization.

# In[ ]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    ExtraTreesClassifier( ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {   'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
)


# In[ ]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[ ]:


modeletfTuned=ExtraTreesClassifier(criterion="entropy",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[ ]:


start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


yetf=modeletfTuned.predict(X_test)


# In[ ]:


yetftest=modeletfTuned.predict(test)


# In[ ]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[ ]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# ## Modelling with KNeighborsClassifier

# In[ ]:


modelneigh = KNeighborsClassifier(n_neighbors=4)


# In[ ]:


start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


classes = modelneigh.predict(X_test)

classes


# In[ ]:


(classes == y_test).sum()/y_test.size 


# In[ ]:


f1 = f1_score(y_test, classes, average='macro')
f1


# ## Performing tuning using Bayesian Optimization.

# In[ ]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    KNeighborsClassifier(
       n_neighbors=4         # No need to tune this parameter value
      ),
    {"metric": ["euclidean", "cityblock"]},
    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
   )


# In[ ]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[ ]:


modelneighTuned = KNeighborsClassifier(n_neighbors=4,
               metric="cityblock")


# In[ ]:


start = time.time()
modelneighTuned = modelneighTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


yneigh=modelneighTuned.predict(X_test)


# In[ ]:


yneightest=modelneighTuned.predict(test)


# In[ ]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[ ]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# ## Modelling with GradientBoostingClassifier

# In[ ]:


modelgbm=gbm()


# In[ ]:


start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


classes = modelgbm.predict(X_test)

classes


# In[ ]:


(classes == y_test).sum()/y_test.size 


# In[ ]:


f1 = f1_score(y_test, classes, average='macro')
f1


# ## Performing tuning using Bayesian Optimization.

# In[ ]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    gbm(
               # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 2                # Number of cross-validation folds
)


# In[ ]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[ ]:


modelgbmTuned=gbm(
               max_depth=84,
               max_features=11,
               min_weight_fraction_leaf=0.04840,
               n_estimators=489)


# In[ ]:


start = time.time()
modelgbmTuned = modelgbmTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


ygbm=modelgbmTuned.predict(X_test)


# In[ ]:


ygbmtest=modelgbmTuned.predict(test)


# In[ ]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[ ]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# ## Modelling with XGBClassifier

# In[ ]:


modelxgb=XGBClassifier()


# In[ ]:


start = time.time()
modelxgb = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


classes = modelxgb.predict(X_test)

classes


# In[ ]:


(classes == y_test).sum()/y_test.size 


# In[ ]:


f1 = f1_score(y_test, classes, average='macro')
f1


# ## Performing tuning using Bayesian Optimization.

# In[ ]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    XGBClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[ ]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[ ]:


modelxgbTuned=XGBClassifier(criterion="gini",
               max_depth=4,
               max_features=15,
               min_weight_fraction_leaf=0.05997,
               n_estimators=499)


# In[ ]:


start = time.time()
modelxgbTuned = modelxgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


yxgb=modelxgbTuned.predict(X_test)


# In[ ]:


yxgbtest=modelxgbTuned.predict(test)


# In[ ]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[ ]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# ## Modelling with Light Gradient Booster

# In[ ]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[ ]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


classes = modellgb.predict(X_test)

classes


# In[ ]:


(classes == y_test).sum()/y_test.size 


# In[ ]:


f1 = f1_score(y_test, classes, average='macro')
f1


# ## Performing tuning using Bayesian Optimization.

# In[ ]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    lgb.LGBMClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[ ]:



# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[ ]:


modellgbTuned = lgb.LGBMClassifier(criterion="gini",
               max_depth=5,
               max_features=53,
               min_weight_fraction_leaf=0.01674,
               n_estimators=499)


# In[ ]:


start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[ ]:


ylgb=modellgbTuned.predict(X_test)


# In[ ]:


ylgbtest=modellgbTuned.predict(test)


# In[ ]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[ ]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# 

#                                             ACCURACY           f1                ACCURACY
#                                 with default parameters       score          with  parameters tuned with Bayesian                                                                                             Optimization 
#         RandomForestClassifier         77.87                   65.52            85.61
#         KNeighborsClassifier           80.70                   72.03            81.85 
#         ExtraTreesClassifier           77.98                   66.47            86.97
#         GradientBoostingClassifier     80.75                   67.03            91.42 
#         XGBoost                        78.03                   61.01            91.57
#         LightGBM                       93.41                   89.43            92.05 

# BUILDING A NEW DATASETS WITH Predicted Values using 6 models

# In[ ]:


NewTrain = pd.DataFrame()
NewTrain['yrf'] = yrf.tolist()
NewTrain['yetf'] = yetf.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['ygbm'] = ygbm.tolist()
NewTrain['yxgb'] = yxgb.tolist()
NewTrain['ylgb'] = ylgb.tolist()

NewTrain.head(5), NewTrain.shape


# In[ ]:


NewTest = pd.DataFrame()
NewTest['yrf'] = yrftest.tolist()
NewTest['yetf'] = yetftest.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ygbm'] = ygbmtest.tolist()
NewTest['yxgb'] = yxgbtest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()
NewTest.head(5), NewTest.shape


# In[ ]:


NewModel=rf(criterion="entropy",
               max_depth=77,
               max_features=6,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[ ]:


start = time.time()
NewModel = NewModel.fit(NewTrain, y_test)
end = time.time()
(end-start)/60


# In[ ]:


ypredict=NewModel.predict(NewTest)


# In[ ]:


ylgbtest


# In[ ]:


submit=pd.DataFrame({'Id': ids, 'Target': ylgbtest})
submit.head(5)


# In[ ]:


submit.to_csv('submit.csv', index=False)

