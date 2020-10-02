#!/usr/bin/env python
# coding: utf-8

# # Costa Rican Household Poverty Level Prediction
# 
# In this notebook, get introduced to the problem, then perform a thorough Exploratory Data Analysis of the dataset, work on feature engineering, try out multiple machine learning models, optimize those models, inspect the outputs of the models, draw conclusions and finally select a model to predict. 
# 
# Data Explanation
# The data is provided in two files: train.csv and test.csv. The training set has 9557 rows and 143 columns while the testing set has 23856 rows and 142 columns. Each row represents one individual and each column is a feature, either unique to the individual, or for the household of the individual. The training set has one additional column, Target, which represents the poverty level on a 1-4 scale and are the labels for this dataset. A value of 1 is the most extreme poverty, 2 = moderate poverty, 3 = vulnerable households and 4 = non vulnerable households
# 
# This is a supervised multi-class classification machine learning problem
# 
# Objective
# The objective is to predict poverty on a household level. Moreover, we have to make a prediction for every individual in the test set, but "ONLY the heads of household are used in scoring" which means we want to predict poverty on a household basis.
# While all members of a household should have the same label in the training data, there are errors where individuals in the same household have different labels. In these cases, we are told to use the label for the head of each household, which can be identified by the rows where parentesco1 == 1.0. 
# 
# Of all 143 columns, few to note are below:
# idhogar: a unique identifier for each household. This variable is not a feature, but will be used to group individuals by household as all individuals in a household will have the same identifier.
# parentesco1: indicates if this person is the head of the household.
# Target: the label, which should be equal for all members in a household
# 
# While making the model, we'll train on a household basis with the label for each household the poverty level of the head of household. The raw data contains a mix of both household and individual characteristics and for the individual data, we will have to find a way to aggregate this for each household. Some of the individuals belong to a household with no head of household which means that unfortunately we can't use this data for training. 
# 
# Roadmap
# The end objective is a machine learning model that can predict the poverty level of a household. We want to evaluate numerous models before choosing one as the "best" and after building a model, we want to investigate the predictions. The roadmap would be as follows :
# 
# 1. Understand the problem 
# 2. Exploratory Data Analysis
# 3. Feature engineering to create a dataset for machine learning
# 4. Compare several machine learning models
# 5. Optimize all the models
# 6. Investigate model predictions in context of problem
# 7. Select the best model
# 8. Draw conclusions and lay out next steps
# 
# Getting Started
# 
# Imports - Data science libraries: Pandas, numpy, matplotlib, seaborn, and eventually sklearn for modeling.

# In[100]:


# 1.1 Load pandas, numpy and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

# Image manipulation
from skimage.io import imshow, imsave

# Image normalizing and compression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif  # Selection criteria
import eli5
from eli5.sklearn import PermutationImportance

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


# ### Loading data and look at Summary Information

# In[101]:


pd.options.display.max_columns = 150

# Read in data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# In[102]:


train.info()


# In[103]:


test.info()


# In[104]:


ids=test['Id']


# In[105]:


train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
plt.xlabel('Number of Unique Values'); plt.ylabel('Count');
plt.title('Count of Unique Values in Integer Columns');


# In[106]:


test.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
plt.xlabel('Number of Unique Values'); plt.ylabel('Count');
plt.title('Count of Unique Values in Integer Columns');


# In[107]:


train.select_dtypes('object').head()


# In[108]:


test.select_dtypes('object').head()


# In[109]:


mapping = {"yes": 1, "no": 0}

# Apply same operation to both train and test
for df in [train, test]:
    # Fill in the values with the correct mapping
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

train[['dependency', 'edjefa', 'edjefe']].describe()


# In[110]:


test[['dependency', 'edjefa', 'edjefe']].describe()


# In[111]:


# Set a few plotting defaults
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'

from collections import OrderedDict

# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

plt.figure(figsize = (16, 12))
plt.style.use('fivethirtyeight')

# Iterate through the float columns
for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):
    ax = plt.subplot(3, 1, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)


# In[112]:


test.shape


# In[113]:


train.shape


# In[114]:


train.shape


# In[115]:


# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[116]:


# Number of missing in each column
missingte = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missingte['percent'] = missingte['total'] / len(test)

missingte.sort_values('percent', ascending = False).head(10)


# In[117]:


# Variables indicating home ownership
own_variables = [x for x in train if x.startswith('tipo')]


# Plot of the home ownership variables for home missing rent payments
train.loc[train['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),
                                                                        color = 'green',
                                                              edgecolor = 'k', linewidth = 2);
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 60)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);


# In[118]:


# Variables indicating home ownership
own_variableste = [x for x in test if x.startswith('tipo')]


# Plot of the home ownership variables for home missing rent payments
test.loc[test['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),
                                                                        color = 'green',
                                                              edgecolor = 'k', linewidth = 2);
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 60)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);


# In[119]:


# Fill in households that own the house with 0 rent payment
train.loc[(train['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
train['v2a1-missing'] = train['v2a1'].isnull()

train['v2a1-missing'].value_counts()


# In[120]:


# Fill in households that own the house with 0 rent payment
test.loc[(test['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
test['v2a1-missing'] = test['v2a1'].isnull()

test['v2a1-missing'].value_counts()


# In[121]:


train.loc[train['rez_esc'].notnull()]['age'].describe()


# In[122]:


test.loc[test['rez_esc'].isnull()]['age'].describe()


# #### Feature Engineering

# Plot Two Categoricals
# 
# We draw a value count plot for where these values missing

# In[123]:


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


# In[124]:


train.shape


# ### Dropping unnecesary columns

# In[125]:


train.drop(['Id','idhogar','v2a1-missing'], inplace = True, axis =1)

test.drop(['Id','idhogar','v2a1-missing'], inplace = True, axis =1)


# In[126]:


train.shape


# In[127]:


test.shape


# ### Dividing the data into predictors & target

# In[128]:


y = train.iloc[:,140]
y.unique()


# In[129]:


X = train.iloc[:,1:141]
X.shape


# ### Scaling  numeric features & applying PCA to reduce features

# In[130]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler as ss

sm_imputer = SimpleImputer()
X = sm_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)
#pca = PCA(0.95)
#X = pca.fit_transform(X)


# In[131]:


X.shape, y.shape,test.shape


# ### Splitting the data into train & test 

# In[132]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# ## Building Models
# 
# ### 1. ExtraTreeClassifier model

# In[133]:


from sklearn.ensemble import ExtraTreesClassifier

modeletf = ExtraTreesClassifier()


# In[134]:


start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[135]:


classes = modeletf.predict(X_test)

classes


# In[136]:


(classes == y_test).sum()/y_test.size


# In[137]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, classes, average='macro')
f1


# ### Tuning using Bayesian optimisation

# In[138]:


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


# In[139]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[140]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[141]:


modeletfTuned=ExtraTreesClassifier(criterion="entropy",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[142]:


start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[143]:


yetf=modeletfTuned.predict(X_test)
yetf


# In[144]:


yetftest=modeletfTuned.predict(test)
yetftest


# In[145]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[146]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# ### 2. GradientBoostingClassifier model

# In[147]:


from sklearn.ensemble import GradientBoostingClassifier as gbm

modelgbm=gbm()


# In[148]:


start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[149]:


classes = modelgbm.predict(X_test)

classes


# In[150]:


(classes == y_test).sum()/y_test.size 


# In[151]:


f1 = f1_score(y_test, classes, average='macro')
f1


# #### Tuning using Bayesian Optimization.

# In[152]:


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


# In[153]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[154]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[155]:


modelgbmTuned=gbm(
               max_depth=84,
               max_features=11,
               min_weight_fraction_leaf=0.04840,
               n_estimators=489)


# In[156]:


start = time.time()
modelgbmTuned = modelgbmTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[157]:


ygbm=modelgbmTuned.predict(X_test)
ygbm


# In[158]:


ygbmtest=modelgbmTuned.predict(test)
ygbmtest


# In[159]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[160]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# ### 3. Light Gradient Booster model

# In[161]:


from lightgbm import LGBMClassifier

modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[162]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[163]:


classes = modellgb.predict(X_test)

classes


# In[164]:


(classes == y_test).sum()/y_test.size 


# In[165]:


f1 = f1_score(y_test, classes, average='macro')
f1


# #### tuning using Bayesian Optimization

# In[166]:


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


# In[167]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[168]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[169]:


modellgbTuned = lgb.LGBMClassifier(criterion="gini",
               max_depth=5,
               max_features=53,
               min_weight_fraction_leaf=0.01674,
               n_estimators=499)


# In[170]:


start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[171]:


ylgb=modellgbTuned.predict(X_test)
ylgb


# In[172]:


ylgbtest=modellgbTuned.predict(test)
ylgbtest


# In[173]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[174]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[175]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# ### 4. Random Forest model

# In[176]:


modelrf = rf()


# In[177]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[178]:


classes = modelrf.predict(X_test)
classes


# In[179]:


(classes == y_test).sum()/y_test.size 


# In[180]:


f1 = f1_score(y_test, classes, average='macro')
f1


# #### tuning using Bayesian Optimisation

# In[181]:


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


# In[182]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[183]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[184]:


modelrfTuned=rf(criterion="entropy",
               max_depth=77,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[185]:


start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[186]:


yrf=modelrfTuned.predict(X_test)
yrf


# In[187]:


yrftest=modelrfTuned.predict(test)
yrftest


# In[188]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[189]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


#                                               ACCURACY           f1                ACCURACY
#                                 with default parameters       score          with  parameters tuned with Bayesian                                                                                             Optimization 
#         ExtraTreesClassifier           0.99                    0.99             1.0
#         GradientBoostingClassifier     1.0                     1.0              1.0 
#         LightGBM                       1.0                     1.0              1.0
#         RandomForestClassifier         1.0                     1.0              1.0

# ## BUILDING NEW DATASETS with predicted values using 4 models

# In[190]:


train1 = pd.DataFrame()

train1['yetf'] = yetf.tolist()
train1['ygbm'] = ygbm.tolist()
train1['ylgb'] = ylgb.tolist()
train1['yrf'] = yrf.tolist()

train1.head(5), train1.shape


# In[191]:


test1 = pd.DataFrame()

test1['yetf'] = yetftest.tolist()
test1['ygbm'] = ygbmtest.tolist()
test1['ylgb'] = ylgbtest.tolist()
test1['yrf'] = yrftest.tolist()

test1.head(5), test1.shape


# In[192]:


EnsembleModel=rf(criterion="entropy",
               max_depth=77,
               max_features=4,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[193]:


start = time.time()
EnsembleModel = EnsembleModel.fit(train1, y_test)
end = time.time()
(end-start)/60


# In[194]:


ypredict=EnsembleModel.predict(test1)


# In[195]:


ypredict


# In[196]:


ygbmtest


# In[197]:


submit=pd.DataFrame({'Id': ids, 'Target': ygbmtest})
submit.head(5)


# In[198]:


submit.to_csv('submit.csv', index=False)

