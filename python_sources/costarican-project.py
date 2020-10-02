#!/usr/bin/env python
# coding: utf-8

# # Costa Rican Household Poverty Level Prediction

# In[ ]:


# Train and test datset was given to work as a project on the poverty level prediction. Source : Kaggle

#Steps to be be followed.
#a. Explore data and perform data visualization
#b. Fill in missing values (NULL values) either using mean or median (if the attribute is numeric) or most-frequently occurring value if the attribute is 'object' or categorical.
#b. Perform feature engineering, may be using some selected features and only from numeric features.
#c. Scale numeric features, AND IF REQUIRED, perform One HOT Encoding of categorical features
#d. IF number of features is very large, please do not forget to do PCA.
#e. Select some estimators for your work.


#Modelling tried below
#       GradientBoostingClassifier
#        RandomForestClassifier
#        KNeighborsClassifier
#        XGBoost
#        LightGBM
        
#        followed by perform tuning using Bayesian Optimization after each modelling.



# In[ ]:


get_ipython().run_line_magic('reset', '-f')


# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[ ]:


from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance


# In[ ]:


import os
import time
import gc
import random
from scipy.stats import uniform
import warnings


# In[ ]:


#path="E:\\bigdata\\costa rica assignment\\costa-rican-household-poverty-prediction\\"
#file_name="train.csv"
#file_path_name =path+file_name
#print(file_path_name)
#train=pd.read_csv(file_path_name)
train=pd.read_csv("../input/train.csv")
print('executing')                  
train.info()


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


#path="E:\\bigdata\\costa rica assignment\\costa-rican-household-poverty-prediction\\"
#file_name="test.csv"
#file_path_name =path+file_name
#print(file_path_name)
#test=pd.read_csv(file_path_name)
test=pd.read_csv("../input/test.csv")
print('executing') 
test.info()


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


ids=test['Id']


# # Plotting using train data

# In[ ]:


sns.countplot("Target", data = train)


# In[ ]:


sns.countplot(x="v2a1",hue="Target",data=train)


# In[ ]:


sns.countplot(x="r4t3",hue="Target",data=train)  #Total persons in the household


# In[ ]:


pd.value_counts(train['Target']).plot.bar()
plt.title('total persons in household histogram')
plt.xlabel('r4t3')
plt.ylabel('count')
train['Target'].value_counts()


# In[ ]:


sns.countplot(x="tamhog",hue="Target",data=train) #tamhog, size of the household


# In[ ]:


sns.countplot(x="tamviv",hue="Target",data=train)#tamviv, number of persons living in the household


# Identify the Object data types to convert "yes" and "no" to 1 and 0

# In[ ]:


train.select_dtypes('object').head()


# In[ ]:


yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)


# In[ ]:


test.select_dtypes('object').head()


# In[ ]:


yes_no_map = {'no':0,'yes':1}
test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)
test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)
test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)


# In[ ]:


train[["dependency","edjefe","edjefa"]].describe()


# In[ ]:


test[["dependency","edjefe","edjefa"]].describe()


# view float64 data type variables

# In[ ]:


train.select_dtypes('float64').head()


# now convert all null "NaN" to  0

# In[ ]:


missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})
missing.sort_values('total', ascending = False).head(10)


# In[ ]:


test.select_dtypes('float64').head()


# In[ ]:


missingT = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})
missingT.sort_values('total', ascending = False).head(10)


# now we will convert these columns rez_esc,v18q1,v2a1,SQBmeaned andcmeaneduc to zero

# In[ ]:


train['rez_esc'] = train['rez_esc'].fillna(0)
train['v18q1'] = train['v18q1'].fillna(0)
train['v2a1'] = train['v2a1'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)


# In[ ]:


test['rez_esc'] = test['rez_esc'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)


# double check if all the above columns are converted

# In[ ]:


missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})
missing.sort_values('total', ascending = False).head(10)


# In[ ]:


missingT = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})
missingT.sort_values('total', ascending = False).head(10)


# Drop columns in train data

# In[ ]:


train.drop(['Id','idhogar'], inplace = True, axis =1)
test.drop(['Id','idhogar'], inplace = True, axis =1)


# target and predictors

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


y = train.iloc[:,140]
y.unique()


# In[ ]:


X = train.iloc[:,1:141]
X.shape


# scaling and PCA

# In[ ]:


my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)


# In[ ]:


X.shape


# In[ ]:


my_imputer = SimpleImputer()
test = my_imputer.fit_transform(test)
scale = ss()
test = scale.fit_transform(test)


# # Begin Data Modelling using Train Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state=1)


# # Modelling using Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier as rf


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


#  Performing tuning using Bayesian Optimization.

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


bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


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


bayes_cv_tuner.best_score_


# In[ ]:


bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


bayes_cv_tuner.cv_results_['params']


# # Modelling with KNeighborsClassifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
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


# Performing tuning using Bayesian Optimization.

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


bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


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


bayes_cv_tuner.best_score_


# In[ ]:


bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


bayes_cv_tuner.cv_results_['params']


# # Modelling with GradientBoostingClassifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier as gbm
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


# Performing tuning using Bayesian Optimization

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


bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


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


bayes_cv_tuner.best_score_


# In[ ]:


bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


bayes_cv_tuner.cv_results_['params']


# # Modelling with XGBClassifier

# In[ ]:


from xgboost.sklearn import XGBClassifier
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


# Performing tuning using Bayesian Optimization.

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


bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


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


bayes_cv_tuner.best_score_


# In[ ]:


bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


bayes_cv_tuner.cv_results_['params']


# # Modelling with Light Gradient Booster

# In[ ]:


import lightgbm as lgb
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


# Performing tuning using Bayesian Optimization.

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


bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


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


bayes_cv_tuner.best_score_


# In[ ]:


bayes_cv_tuner.score(X_test, y_test)


# In[ ]:


bayes_cv_tuner.cv_results_['params']


# In[ ]:


#                                                     Summary
#                                 ACCURACY                   f1            ACCURACY
#                            with default parameters       score          with  parameters tuned with Bayesian                                                                                             Optimization 
#    RandomForestClassifier         0.96                    0.93              1.0
#    KNeighborsClassifier           0.81                    0.70              0.77 
#    GradientBoostingClassifier     1.0                     1.0               1.0 
#    XGBoost                        1.0                     1.0               1.0
#    LightGBM                       1.0                     1.0               1.0


# In[ ]:


NewTrain = pd.DataFrame()
NewTrain['yrf'] = yrf.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['ygbm'] = ygbm.tolist()
NewTrain['yxgb'] = yxgb.tolist()
NewTrain['ylgb'] = ylgb.tolist()

NewTrain.head(5), NewTrain.shape


# In[ ]:


NewTest = pd.DataFrame()
NewTest['yrf'] = yrftest.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ygbm'] = ygbmtest.tolist()
NewTest['yxgb'] = yxgbtest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()

NewTest.head(5), NewTest.shape


# In[ ]:


NewModel=rf(criterion="entropy",
               max_depth=77,
               max_features=5,
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


# In[ ]:


#submit.to_csv('E:/bigdata/costa rica assignment/Nagendrafinalsubmit.csv', index=False)

