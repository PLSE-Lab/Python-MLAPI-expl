#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

get_ipython().run_line_magic('reset', '-f')


#  **CALLING REQUIRED LIBRARIES**

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#FOR DATA PROCESSING
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

#FOR MODELING ESTIMATORS
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

#FOR MEASURING PERFORMANCE
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#FOR TUNING PARAMETERS
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance

#MISC
import os
import time
import gc
import random
from scipy.stats import uniform
import warnings


# **A. Explore data and perform data visualization**

# In[3]:


#Reading the dataset for training the model and validating it.
train=pd.read_csv("../input/train.csv")

#Reading the final Test dataset on which the predictions are done
test=pd.read_csv("../input/test.csv")
print ("Train Dataset: Rows, Columns: ", train.shape)
print ("Test Dataset: Rows, Columns: ", test.shape)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train.shape, test.shape


# In[7]:


train.info() 


# In[8]:


ids=test['Id']


# **Feature-Target Relationships**

# **1. Different Levels of Poverty Household Groups**

# In[9]:


sns.countplot("Target", data=train)
plt.title('Poverty Levels')


# In[10]:


sns.countplot(x="r4t3",hue="Target",data=train)


# In[11]:


sns.countplot(x="hhsize",hue="Target",data=train)


# In[12]:


sns.countplot(x="v18q1",hue="Target",data=train)


# In[13]:


sns.countplot(x="tamhog",hue="Target",data=train)


# In[14]:


sns.countplot(x="abastaguano",hue="Target",data=train)


# In[15]:


sns.countplot(x="noelec",hue="Target",data=train)


# **B. Checking for Missing Value****

# In[16]:


train_null = train.isnull().sum()
train_null_non_zero = train_null[train_null>0] / train.shape[0]


# In[17]:


train_null_non_zero


# In[18]:


sns.barplot(x=train_null_non_zero, y=train_null_non_zero.index)
_ = plt.title('Fraction of NaN values, %')


# In[19]:


train.select_dtypes('object').head()


# **Fill in missing values (NULL values) using 1 for yes and 0 for no**

# In[20]:


yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)


# In[21]:


#yes_no_map = {'no':0,'yes':1}
test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)
test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)
test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)


# **Converting categorical objects into numericals**

# In[22]:


train[["dependency","edjefe","edjefa"]].describe()


# In[23]:


train[["dependency","edjefe","edjefa"]].hist()


# **Fill in missing values (NULL values) using 1 for yes and 0 for no**

# In[24]:


# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[25]:


train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)
train['v2a1'] = train['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)
train['rez_esc'] = train['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)


# **Dropping unnecesary columns**

# In[26]:


train.drop(['Id','idhogar'], inplace = True, axis =1)
test.drop(['Id','idhogar'], inplace = True, axis =1)


# In[27]:


train.shape, test.shape


# **Dividing the data into predictors & target**

# In[28]:


y = train.iloc[:,140]
y.unique()


# In[29]:


X = train.iloc[:,1:141]
X.shape


# **Splitting the data into Train and Test**

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# **MODELING
# 
# **Modelling with GradientBoostingClassifier****

# In[31]:


modelgbm=gbm()


# In[32]:


start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[33]:


classes = modelgbm.predict(X_test)
classes


# In[34]:


(classes == y_test).sum()/y_test.size 


# **Performing tuning using Bayesian Optimization**

# In[35]:


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


# In[36]:


# Optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[37]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[38]:


modelgbmTuned=gbm(
               max_depth=31,
               max_features=29,
               min_weight_fraction_leaf=0.02067,
               n_estimators=489)


# In[39]:


start = time.time()
modelgbmTuned = modelgbmTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[40]:


ygbm=modelgbmTuned.predict(X_test)
ygbmtest=modelgbmTuned.predict(test)


# In[41]:


#  Acheived average accuracy during cross-validation
bayes_cv_tuner.best_score_


# In[42]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[43]:


#  All sets of parameters that were tried
bayes_cv_tuner.cv_results_['params']


# ***MODELLING WITH RandomForestClassifier**

# In[44]:


modelrf = rf()


# In[45]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[46]:


classes = modelrf.predict(X_test)


# In[47]:


(classes == y_test).sum()/y_test.size


# **Performing tuning using Bayesian Optimization.**

# In[48]:


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


# In[49]:


# Optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[50]:


#  Best-parameters list
bayes_cv_tuner.best_params_


# In[51]:


modelrfTuned=rf(criterion="entropy",
               max_depth=77,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[52]:


start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[53]:


yrf=modelrfTuned.predict(X_test)
yrftest=modelrfTuned.predict(test)


# In[54]:


# Acheived average accuracy during cross-validation
bayes_cv_tuner.best_score_


# In[55]:


#  Available accuracy on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[56]:


#  All sets of parameters that were tried
bayes_cv_tuner.cv_results_['params']


# ***MODELLING WITH KNeighborsClassifier**

# In[57]:


modelneigh = KNeighborsClassifier(n_neighbors=7)


# In[58]:


start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[59]:


classes = modelneigh.predict(X_test)
classes


# In[60]:


(classes == y_test).sum()/y_test.size 


# **Performing tuning using Bayesian Optimization.**

# In[61]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    KNeighborsClassifier(
       n_neighbors=7         # No need to tune this parameter value
      ),
    {"metric": ["euclidean", "cityblock"]},
    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
   )


# In[62]:


# Optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[63]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[64]:


modelneighTuned = KNeighborsClassifier(n_neighbors=7,
               metric="cityblock")


# In[65]:


start = time.time()
modelneighTuned = modelneighTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[66]:


yneigh=modelneighTuned.predict(X_test)


# In[67]:


yneightest=modelneighTuned.predict(test)


# In[68]:


# Acheived average accuracy during cross-validation
bayes_cv_tuner.best_score_


# In[69]:


#  Available accuracy on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[70]:


#  All sets of parameters that were tried
bayes_cv_tuner.cv_results_['params']


# **Modelling with ExtraTreeClassifier**

# In[71]:


modeletf = ExtraTreesClassifier()


# In[72]:


start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[73]:


classes = modeletf.predict(X_test)
classes


# In[74]:


(classes == y_test).sum()/y_test.size


# **Performing tuning using Bayesian Optimization.**

# In[75]:


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


# In[76]:


# Optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[77]:


#  Best-parameters list
bayes_cv_tuner.best_params_


# In[78]:


modeletfTuned=ExtraTreesClassifier(criterion="entropy",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=100)


# In[79]:


start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[80]:


yetf=modeletfTuned.predict(X_test)
yetftest=modeletfTuned.predict(test)


# In[81]:


#  Acheived average accuracy during cross-validation
bayes_cv_tuner.best_score_


# In[82]:


#  Available accuracy on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[83]:


#  All sets of parameters that were tried
bayes_cv_tuner.cv_results_['params']


# **Modelling with XGBoosterClassifier**

# In[84]:


modelxgb=XGBClassifier()


# In[85]:


start = time.time()
modelxgb = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[86]:


classes = modelxgb.predict(X_test)
classes


# In[87]:


(classes == y_test).sum()/y_test.size 


# **Performing tuning using Bayesian Optimization.**

# In[88]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    XGBClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 300),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[89]:


# Optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[90]:


#  Best-parameters list
bayes_cv_tuner.best_params_


# In[91]:


modelxgbTuned=XGBClassifier(criterion="gini",
               max_depth=85,
               max_features=47,
               min_weight_fraction_leaf=0.035997,
               n_estimators=178)


# In[92]:


start = time.time()
modelxgbTuned = modelxgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[93]:


#  Acheived average accuracy during cross-validation
bayes_cv_tuner.best_score_


# In[94]:


#  Available accuracy on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[95]:


#  All sets of parameters that were tried
bayes_cv_tuner.cv_results_['params']


# **Modelling with Light Gradient Booster**

# In[96]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[97]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[98]:


classes = modellgb.predict(X_test)
classes


# In[99]:


(classes == y_test).sum()/y_test.size 


# **Performing tuning using Bayesian Optimization.**

# In[100]:


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


# In[101]:


# Optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[102]:


#  Best-parameters list
bayes_cv_tuner.best_params_


# In[103]:


modellgbTuned = lgb.LGBMClassifier(criterion="entropy",
               max_depth=35,
               max_features=14,
               min_weight_fraction_leaf=0.18611,
               n_estimators=148)


# In[104]:


start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[105]:


ylgb=modellgbTuned.predict(X_test)
ylgbtest=modellgbTuned.predict(test)


# In[106]:


#  Acheived average accuracy during cross-validation
bayes_cv_tuner.best_score_


# In[107]:


#  Available accuracy on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[108]:


#  All sets of parameters that were tried
bayes_cv_tuner.cv_results_['params']


# **Building new dataset with predicted results by combining all these models**

# In[109]:


NewTrain = pd.DataFrame()
NewTrain['yetf'] = yetf.tolist()
NewTrain['yrf'] = yrf.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['ygbm'] = ygbm.tolist()
NewTrain['ylgb'] = ylgb.tolist()

NewTrain.head(5), NewTrain.shape


# In[110]:


NewTest = pd.DataFrame()
NewTest['yetf'] = yetftest.tolist()
NewTest['yrf'] = yrf.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ygbm'] = ygbmtest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()

NewTest.head(5), NewTest.shape


# In[111]:


NewModel=rf(criterion="entropy",
               max_depth=87,
               max_features=4,
               min_weight_fraction_leaf=0.0,
               n_estimators=600)


# In[112]:


start = time.time()
NewModel = NewModel.fit(NewTrain, y_test)
end = time.time()
(end-start)/60


# In[113]:


ypredict=NewModel.predict(NewTest)
ypredict


# In[114]:


#submit=pd.DataFrame({'Id': ids, 'Target': ylgbtest})
submit=pd.DataFrame({'Id': ids, 'Target': ypredict})
submit.head(5)


# In[115]:


submit.to_csv('submit.csv', index=False)


# In[116]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = ypredict
sub.drop(sub.columns[[1]], axis=1, inplace=True)
sub.to_csv('submission.csv',index=False)

