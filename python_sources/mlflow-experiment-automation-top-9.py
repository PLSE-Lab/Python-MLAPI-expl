#!/usr/bin/env python
# coding: utf-8

# # How automate and log your experiments

# ## Automated experiment tracking with MLflow
# 
# During the competition I did many experiments and quit often I submited results with some best score and after thet can't reproduce it. So, I used my personal help tool set:
# - Git repository. After submission I commit code that generate that submission file
# - Local log file to record all usefull information in parallel with printf
# - And MLflow. It help me to compare many different experiments and did aditional analyse on how feature engeeniring impact model perforamnce
# 
# Official site: https://mlflow.org/
# 
# Instalation tutorial: https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
# 
# - But you can quickly install: **pip install mlflow**
# - After installation go to folder with this notebook and run cmd **mlflow ui** in terminal
# - It will run server at http://localhost:5000
# - More examples here: https://github.com/mlflow/mlflow/tree/master/examples
# 
# TIP: Some times mlflow could fail, so just run **mlflow.end_run()**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import optuna

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ElasticNet, 
    Lasso,
    LinearRegression,
    Ridge
)
from scipy.stats import norm, skew, boxcox_normmax #for some statistics
from sklearn import utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE, f_regression
from scipy.special import boxcox1p

import datetime
import time


# ## Import mlflow
# - First of all you should create your new experiment. 
# - Call get_tracking_uri return you a path to local folder with meta files/results of your experiments
# - Call get_artifact_uri return path to your artifacts

# In[ ]:


# Note:  We can't set this here due to https://github.com/mlflow/mlflow/issues/608
#tracking_uri='file:///mnt/pipelineai/users/experiments'
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import mlflow.xgboost

experiment_name = 'house_price'
mlflow.set_experiment(experiment_name)
# Forcing an end_run() to prevent 
#    https://github.com/mlflow/mlflow/issues/1335 
#    https://github.com/mlflow/mlflow/issues/608
mlflow.end_run()

artifact_path = mlflow.get_artifact_uri()
uri = mlflow.tracking.get_tracking_uri()
print(artifact_path)
print(uri)


# - We will TAGs for more easy filtering between different models
# - Parameters help us to track training configuration / conditions
# - Metrics helps us filter models by score
# - Log_model will save model to the artifacts

# In[ ]:


def log_mlflow(model):
    # Track params and metrics 
    with mlflow.start_run() as run:
        mlflow.set_tag("model_name", name)
        mlflow.log_param("CV_n_folds", CV_n_folds)
        mlflow.log_param("TEST_PART", TEST_PART)
        mlflow.log_param("Train size", X_train.shape)
        mlflow.log_param("Colums", str(X_train.columns.values.tolist()))
        mlflow.log_metrics({'rmse_cv': score_cv.mean(), 'rmse': score})
        mlflow.log_metric("rmse_train", score_train)
        # Save model to artifacts
        mlflow.sklearn.log_model(model, name)
    mlflow.end_run()


# If your code crashed and you didn't finish mlflow you shoud run the following code:

# In[ ]:


mlflow.end_run()


# ## Load data 

# In[ ]:


# Import data
path =  '/kaggle/input/house-prices-advanced-regression-techniques/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
print ("Size ntrain= {} / ntest = {}".format(train.shape, test.shape))
# train.head()


# # Features engineering

# ## Sales Price 

# The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

# In[ ]:


# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])


# ## Remove outliers

# In[ ]:


# Deleting outliers
train = train.drop(train[(train['GrLivArea']>4600)].index)
train = train.drop(train[(train['TotalBsmtSF']>5900)].index)
train = train.drop(train[(train['1stFlrSF']>4000)].index)
train = train.drop(train[(train['MasVnrArea']>1500)].index)
train = train.drop(train[(train['GarageArea']>1230)].index)
train = train.drop(train[(train['TotRmsAbvGrd']>13)].index)


# ### Group train and test datasets

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
print ("Size ntrain= {} / ntest = {}".format(ntrain, ntest))
# all_data = pd.concat((train, test)).reset_index(drop=True)
all_data = train.append(test, sort=False).reset_index(drop=True)
#To save original ID for final submission
orig_test = test.copy() 
log_y_train = train['SalePrice']
# To avoid normalization of SalesPrice - drop it from All data 
all_data.drop(['SalePrice'], axis=1, inplace=True)
# Id no need for traning
all_data.drop(['Id'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ### Show statistic how much data is missing

# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# ### Data Correlation

# In[ ]:


threshold = 0.90
# Absolute value correlation matrix
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# upper.head(50)
# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d features to remove.' % (len(collinear_features)))
print(collinear_features)


# ### Imputing missing values

# In[ ]:


# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
# Raplace null with None value.
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data = all_data.drop(['Utilities'], axis=1)


# ### Generate new features

# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# ## Categorial and quantative features investigation

# ### Convert str in Quality features to int

# In[ ]:


def convert_str_to_int(data, features, score):
    all_data[features] = all_data[features].applymap(lambda s: score.get(s) if s in score else s)

featuresQualCond = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual",
                    "FireplaceQu", "HeatingQC", "GarageQual", "GarageCond", "PoolQC"]
qual_score_QualCond = {"None":0, "NA":1, "Po":2, "Fa":3, "TA":4, "Gd":5, "Ex":6}
convert_str_to_int(all_data, featuresQualCond, qual_score_QualCond)

featuresExposure = ["BsmtExposure"]
qual_score = {"None":0, "NA":1, "No":2, "Mn":3, "Av":4, "Gd":5}
convert_str_to_int(all_data, featuresExposure, qual_score)

featuresFinType = ["BsmtFinType1", "BsmtFinType2"]
qual_score = {"None":0, "NA":1, "Unf":2, "LwQ":3, "Rec":4, "BLQ":5, "ALQ":6, "GLQ":7}
convert_str_to_int(all_data, featuresFinType, qual_score)

featuresGarageFin = ["GarageFinish"]
qual_score = {"None":0, "NA":1, "Unf":2, "RFn":3, "Fin":4}
convert_str_to_int(all_data, featuresGarageFin, qual_score)


# ### Some features should be categorial

# In[ ]:


#MSSubClass=The building class represented as int but it's category
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# ## Label Encoding 
# 
# ### some categorical variables that may contain information in their ordering set

# In[ ]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# ## Skewed features

# **Box Cox Transformation of (highly) skewed features**

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skew_features = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("Skew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skew_features})

skewness = skewness[abs(skew_features) > 0.05]
print("There are {} high skewed numerical features to Box Cox transform".format(skewness.shape[0]))

skewed_features = skewness.index
for i in skewed_features:
    all_data[i] = boxcox1p(all_data[i], 0.15)
print('Shape all_data: {}'.format(all_data.shape))


# In[ ]:


# Getting Dummies from Condition1 and Condition2
conditions = set([x for x in all_data['Condition1']] + [x for x in all_data['Condition2']])
dummies = pd.DataFrame(data=np.zeros((len(all_data.index), len(conditions))),
                       index=all_data.index, columns=conditions)
for i, cond in enumerate(zip(all_data['Condition1'], all_data['Condition2'])):
#     dummies.ix[i, cond] = 1
    dummies.ix[i, cond] = 1
all_data = pd.concat([all_data, dummies.add_prefix('Condition_')], axis=1)
all_data.drop(['Condition1', 'Condition2'], axis=1, inplace=True)


# ## convert categorical variable into dummy

# In[ ]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[ ]:


cols = ('Condition_RRNn', 'Condition_RRAe', 
        'Condition_Artery', 'Condition_Feedr', 'Condition_Feedr', 'Condition_RRNe', 
        'Condition_PosA', 'Condition_Norm', 'Condition_RRAn', 
        'Condition_PosN')
# process columns, apply LabelEncoder to categorical features
lbl = LabelEncoder()
for c in cols:
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(all_data[c].values)       
print('Shape all_data: {}'.format(all_data.shape))


# ## Drop low importance features

# In[ ]:


todrop = ['GarageType_BuiltIn', 'MiscFeature_None', 'PoolQC', 'MiscVal', 'Condition_RRAn', 'Neighborhood_BrDale', 'GarageType_2Types', 'Exterior1st_Stucco', 'Neighborhood_Blmngtn', 'LotConfig_FR3', 'Neighborhood_Timber', 'SaleType_ConLI', 'Condition_PosA', 'LandContour_Bnk', 'Alley_None', 'Street_Pave', 'Street_Grvl', 'Condition_Norm', 'Condition_RRNn', '3SsnPorch', 'BldgType_TwnhsE', 'RoofMatl_Membran', 'RoofMatl_WdShake', 'RoofMatl_Roll', 'RoofMatl_Metal', 'Exterior2nd_Stone', 'Exterior2nd_MetalSd', 'MasVnrType_None', 'Exterior2nd_ImStucc', 'LowQualFinSF', 'RoofMatl_Tar&Grv', 'Exterior2nd_AsphShn', 'Heating_GasA', 'HouseStyle_2.5Unf', 'Exterior2nd_AsbShng', 'Exterior1st_WdShing', 'BldgType_Duplex', 'Exterior2nd_CBlock', 'SaleType_Oth', 'Condition_PosN', 'Neighborhood_Veenker', 'BldgType_2fmCon', 'MiscFeature_TenC', 'Neighborhood_Blueste', 'RoofStyle_Mansard', 'Foundation_Slab', 'HouseStyle_SFoyer', 'Heating_Floor', 'HouseStyle_2.5Fin', 'Exterior1st_Stone', 'Exterior1st_CBlock']
print ("Before Drop = ", all_data.shape)
all_data.drop(todrop, axis=1, inplace=True)
print ("After Drop = ", all_data.shape)


# ## Normalize data

# In[ ]:


scaler = RobustScaler()
df_all = pd.DataFrame(scaler.fit_transform(all_data))


# ## Separate back train and test set

# In[ ]:


# all_data = all_data.iloc[:, 1:10]
# data.iloc[:, 0:2] # first two columns of data frame with all rows
train = all_data[:ntrain]
test = all_data[ntrain:]
colnames = train.columns

# Check remaining missing values if any 
train_na = (train.isnull().sum() / len(all_data)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head()


# Before start training log into file all features that we will use for training. Some feature droped due to feature importance to improve model performance. So it nice to have. 
# We will use simple local file for this purpose. 

# In[ ]:


# Log into a file train shape and columns names
f = open("models_training_log.txt", "a+")
print("\n-------------------" + str(datetime.datetime.now().isoformat()) + "-------------------", file=f)
print("Train shape:" + str(train.shape) , file=f)
print("feature names:" + str(list(colnames)) , file=f)
f.close()


# # ====== Train model ======

# In[ ]:


def rmsle_cv(model, X_train, y_train, cv_n_folds):
    kf = KFold(cv_n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def print_rmse_score(y, y_pred):
    score = rmse(y, y_pred)
    print("RMSE score: {:.8f}".format(score))
    return score

def print_rmse_cv_score(model, X_train, y_train, cv_n_folds, prefix=""):
    score = rmsle_cv(model, X_train, y_train, cv_n_folds)
    print(prefix + "CV RMSE score: {:.8f} ({:.4f})".format(score.mean(), score.std()))
    return score

def prepare_datasets(X_matrix, Y_vector, test_part):
    X_train = X_matrix
    y_train = Y_vector
    X_test = np.array([])
    y_test = np.array([])
    if test_part > 0:
        X_train, X_test, y_train, y_test = train_test_split(X_matrix, Y_vector, random_state = 0, test_size = test_part)
            
    print ("\nTEST_PART = ",  test_part)
    print ("Train X| " + str(X_train.shape) + " Y| " + str(y_train.shape))
    print ("Test X| " + str(X_test.shape) + " Y| " + str(y_test.shape))
    return [X_train, y_train, X_test, y_test]


# In[ ]:


X = train.copy()
Y = log_y_train.copy()
X_submit = test.copy()

TEST_PART = 0.1
CV_n_folds = 3
[X_train, y_train, X_test, y_test] = prepare_datasets(X, Y, TEST_PART)


# # -------  Models  ---------
# 

# In[ ]:


classifiers=set([])
models={}
scores_cv={}
scores={}
scores_train={}
submits={}


# In[ ]:


# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# ## Ridge

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_ridge = Ridge(alpha=5.75)\nmodel = model_ridge.fit(X_train, y_train)\nscore_cv = score_ridge_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_ridge = print_rmse_score(y_test, model.predict(X_test))\nridge_submit = np.expm1(model.predict(X_submit))\n\nname = "Ridge"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = ridge_submit\nranks[name] = ranking(np.abs(model.coef_), colnames)\n\nlog_mlflow(model)')


# ## GradientBoostingRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = {\n    \'learning_rate\': 0.01,\n    "n_estimators":500,\n    \'max_depth\': 3,\n    \'max_features\': "sqrt",\n    "loss":"huber",\n    \'min_samples_leaf\': 12,\n    \'min_samples_split\': 11,\n    "random_state":5\n}\nmodel_GBoost = GradientBoostingRegressor(**params)\nmodel = model_GBoost.fit(X_train, y_train)\nscore_cv = score_GBoost_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_GBoost = print_rmse_score(y_test, model.predict(X_test))\ngboost_submit = np.expm1(model.predict(X_submit))\n\nname = "GBoost"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = gboost_submit\nranks[name] = ranking(np.abs(model.feature_importances_), colnames)\n\nlog_mlflow(model)')


# ## KernelRidge

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_KRR = KernelRidge(alpha=0.03525, \n                        kernel=\'polynomial\', \n                        degree=1, coef0=1e-6)\nmodel = model_KRR.fit(X_train, y_train)\nscore_cv = score_KRR_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_KRR = print_rmse_score(y_test, model.predict(X_test))\nkrr_submit = np.expm1(model.predict(X_submit))\n\nname = "KRR"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = krr_submit\n\nlog_mlflow(model)')


# ## LightGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = {\'learning_rate\': 0.01, \'num_leaves\': 3, \'max_bin\': 84,\n          \'bagging_freq\': 1, \'bagging_seed\': 2, \'feature_fraction_seed\': 97, \n          \'bagging_fraction\': 0.745, "verbose":-1,\n          \'objective\': \'regression\',"n_estimators":1000\n         }\nmodel_lgbm = lgb.LGBMRegressor(**params)\nmodel = model_lgbm.fit(X_train, y_train)\nscore_cv = score_lgbm_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_lgbm = print_rmse_score(y_test, model.predict(X_test))\nlgbm_submit = np.expm1(model.predict(X_submit))\n\nname = "lgbm"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = lgbm_submit\nranks[name] = ranking(np.abs(model.feature_importances_), colnames)\nlog_mlflow(model)')


# ## XGBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_xgb = xgb.XGBRegressor(tree_method="hist",\n                             colsample_bytree=0.4603, gamma=0.01468, \n                             learning_rate=0.05187, max_depth=3, \n                             min_child_weight=0.0817, n_estimators=200,\n                             reg_alpha=0.4640, reg_lambda=0.6571,\n                             subsample=0.5213, silent=1,\n                             random_state =7, nthread = -1)\nmodel = model_xgb.fit(X_train, y_train)\nscore_cv = score_xgb_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_xgb = print_rmse_score(y_test, model.predict(X_test))\nxgb_submit = np.expm1(model.predict(X_submit))\n\nname = "xgb"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = xgb_submit\nranks[name] = ranking(np.abs(model.feature_importances_), colnames)\n\nlog_mlflow(model)')


# ## Lasso

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.00055419, random_state=1, max_iter=50000))\nmodel = model_lasso.fit(X_train, y_train)\nscore_cv = score_lasso_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_lasso = print_rmse_score(y_test, model.predict(X_test))\nlasso_submit = np.expm1(model.predict(X_submit))\n\nname = "lasso"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = lasso_submit\nranks[name] = ranking(np.abs(model[1].coef_), colnames)\n\nlog_mlflow(model)')


# ## ElasticNet

# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = {\n    "alpha":0.00185,\n    "max_iter":10000,\n    "l1_ratio":0.224,\n    "random_state":1\n}\nmodel_ENet = make_pipeline(RobustScaler(), ElasticNet(**params))\nmodel = model_ENet.fit(X_train, y_train)\nscore_cv = score_ENet_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_ENet = print_rmse_score(y_test, model.predict(X_test))\nENet_submit = np.expm1(model.predict(X_submit))\n\nname = "ENet"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = ENet_submit\nranks[name] = ranking(np.abs(model[1].coef_), colnames)\n\nlog_mlflow(model)')


# ## Model Stacking

# In[ ]:


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)
        return self

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append(np.exp(regressor.predict(X).ravel()))

        return np.log1p(np.mean(self.predictions_, axis=0))


# In[ ]:


# Averaged base models class
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_averaged = AveragingModels(models = (model_ENet, \n                                            model_lasso, \n                                            model_GBoost,\n                                            model_KRR))\n\nmodel = model_averaged.fit(X_train, y_train)\nscore_cv = score_averaged_cv = print_rmse_cv_score(model, X_train, y_train, CV_n_folds)\nscore = score_averaged = print_rmse_score(y_test, model.predict(X_test))\naveraged_models_submit = np.expm1(model.predict(X_submit))\n\nname = "Averaged"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = averaged_models_submit\n\nlog_mlflow(model)')


# In[ ]:


# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=False, random_state=111)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_stacked_averaged = StackingAveragedModels(base_models = (model_ENet, model_GBoost, model_KRR),\n                                                 meta_model = model_lasso)\n\nmodel = model_stacked_averaged.fit(X_train.values, y_train.values)\nscore_cv = score_stacked_cv = print_rmse_cv_score(model, X_train, y_train.values, CV_n_folds)\nscore = score_stacked = print_rmse_score(y_test, model.predict(X_test.values))\nstacked_averaged_models_submit = np.expm1(model.predict(X_submit.values))\n\nname = "Stacked"\nclassifiers.add(name)\nmodels[name] = model \nscores_cv[name] = score_cv\nscores[name] = score\nscores_train[name] = score_train = print_rmse_score(y_train, model.predict(X_train))\nsubmits[name] = stacked_averaged_models_submit\n\nlog_mlflow(model)')


# # All model comparison owerview
# In this section compare model perforamnce of singe notebook run.

# In[ ]:


CV_score=[]
train_score=[]
Pred_score=[]
score_list=[]
log_y_test = np.expm1(y_test)
y_pred_df = pd.DataFrame({"test":log_y_test}).reset_index(drop=True)
std=[]

for i, name in enumerate(classifiers):
    model = models[name]
    if name != "Stacked":
        y_pred = np.expm1(model.predict(X_test))
    else:
        y_pred = np.expm1(model.predict(X_test.values))
    CV_score.append(scores_cv[name].mean())
    std.append(scores_cv[name].std())
    Pred_score.append(scores[name])
    train_score.append(scores_train[name])
    score_list.append(scores_cv[name])
    
    new_y_pred = pd.DataFrame({name:y_pred})
    y_pred_df = pd.concat([y_pred_df, new_y_pred], axis=1).reset_index(drop=True)
    y_pred_df["del_"+name] = y_pred_df["test"] - y_pred_df[name]
   
    
new_models_dataframe2=pd.DataFrame({"RMSE":Pred_score, "Train - RMSE":train_score,'CV Mean':CV_score,'Std':std}, index=classifiers) 
print("\nTEST_PART=", TEST_PART)
print("CV_n_folds=", CV_n_folds)
print(new_models_dataframe2)
print("\nMin RMSE score: {:.5f}".format(new_models_dataframe2["RMSE"].min()))
print("Min CV mean score: {:.5f}".format(new_models_dataframe2["CV Mean"].min()))

# Log into a file iteration score
f = open("models_training_log.txt", "a")
print("TEST_PART=", TEST_PART, file=f)
print("CV_n_folds=", CV_n_folds, file=f)
print(new_models_dataframe2, file=f)
print("\nMin RMSE score: {:.5f}".format(new_models_dataframe2["RMSE"].min()), file=f)
print("Min CV mean score: {:.5f}".format(new_models_dataframe2["CV Mean"].min()), file=f)
f.close()


# In[ ]:


if CV_n_folds > 2:
    plt.subplots(figsize=(12,6))
    sns.boxplot(list(classifiers), score_list)


# ## Feature importance

# In[ ]:


# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")


# In[ ]:


meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')


# In[ ]:


todrop = list(meanplot[meanplot["Mean Ranking"]<0.02]["Feature"])
print ("Features to drop: {}".format(len(todrop)))
print (todrop)


# ### Show how real prise from Test set corelated with Predicted price by models

# In[ ]:


plt.figure(figsize=(16, 8))
# multiple line plot
plt.plot(y_pred_df.index,'test', data=y_pred_df, marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=4)

colors = ['red', 'yellow', 'green', 'black', 'cyan', 'magenta', 'black', 'green', 'black', 'cyan', 'magenta', 'black']
for i, name in enumerate(classifiers):
    plt.plot(y_pred_df.index, name, data=y_pred_df, marker='o', markerfacecolor=colors[i], markersize=3)

plt.legend()


# Show just selected models

# In[ ]:


plt.figure(figsize=(16, 8))
# multiple line plot
plt.plot(y_pred_df.index,'test', data=y_pred_df, marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=4)
# plt.plot(y_pred_df.index, "lasso", data=y_pred_df, marker='o', markerfacecolor="red", markersize=3)
plt.plot(y_pred_df.index, "Stacked", data=y_pred_df, marker='o', markerfacecolor="green", markersize=3)
# plt.plot(y_pred_df.index, "xgb", data=y_pred_df, marker='X', markerfacecolor="yellow", markersize=6)
plt.plot(y_pred_df.index, "lgbm", data=y_pred_df, marker='X', markerfacecolor="yellow", markersize=6)

plt.legend()


# ### Show delta between real price and predicted

# In[ ]:


plt.figure(figsize=(16, 8))
# multiple line plot
# plt.plot(y_pred_df.index,'test', data=y_pred_df, marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=4)
plt.plot(y_pred_df.index, "del_Averaged", data=y_pred_df, marker='o', markerfacecolor="red", markersize=3)
plt.plot(y_pred_df.index, "del_Stacked", data=y_pred_df, marker='o', markerfacecolor="green", markersize=3)
# plt.plot(y_pred_df.index, "del_xgb", data=y_pred_df, marker='X', markerfacecolor="yellow", markersize=6)
plt.plot(y_pred_df.index, "del_lgbm", data=y_pred_df, marker='X', markerfacecolor="yellow", markersize=6)
# plt.plot(y_pred_df.index, "del_lasso", data=y_pred_df, marker='X', markerfacecolor="red", markersize=6)

plt.legend()


# ### Let's try to see the biggest missprediction

# In[ ]:


fail_idx = y_pred_df[abs(y_pred_df["del_xgb"]) > 60000].index
y_pred_df[abs(y_pred_df["del_xgb"]) >60000]


# In[ ]:


df = X.iloc[fail_idx, :]
df[['OverallQual', 'MasVnrArea', 'TotalSF', '1stFlrSF', '2ndFlrSF',  
    'LotArea', 'LotFrontage', 'GrLivArea', "TotRmsAbvGrd"]]


# In[ ]:


print(y_pred_df["del_Stacked"].mean())
print(y_pred_df["del_xgb"].mean())
print(y_pred_df["del_lgbm"].mean())


# # Generate Submit file

# In[ ]:


def blend_models_predict(names, coeff, X):
    pred = 0;
    for i, name in enumerate(names):
        if type(models[name]) == StackingAveragedModels:
            pred += coeff[i] * models[name].predict(X.values)
        else:
            pred += coeff[i] * models[name].predict(X)
              
    return pred


# In[ ]:


def log_to_file_stackconfig(models, coeff, rmse, rmse_train):
    f = open("models_training_log.txt", "a")
    print("Stacking config:" +str(models)+" "+str(coeff), file=f)
    print("TEST={:0.5f}".format(rmse) + " TRAIN={:0.5f}".format(rmse_train), file=f)
    f.close()


# In[ ]:


compare_models = ["Stacked",
                  "Averaged", 
#                   "xgb", 
                  "lgbm"
                 ]
coeff = [0.35, 0.35, 0.3]

y_train_pred = blend_models_predict(compare_models, coeff, X_train)
y_test_pred = blend_models_predict(compare_models, coeff, X_test)
# Final result convert from log 
Y_pred = np.expm1(blend_models_predict(compare_models, coeff, X_submit))

print("[TRAIN]:")
train_rmse_ = print_rmse_score(y_train, y_train_pred)
print("[TEST]:")
test_score = print_rmse_score(y_test, y_test_pred)
log_to_file_stackconfig(compare_models, coeff, test_score, train_rmse_)

date = datetime.datetime.now().isoformat()
submit_name = str(int(round(time.time() * 1000)))
for i, name in enumerate(compare_models):
    submit_name += "_"+str(coeff[i])+"_" + name 
submit_name += "_TEST_" + str(TEST_PART) 
submit_name += "_rmse_{:0.5f}".format(test_score) + ".csv"
print("="*80)
submission = pd.DataFrame({
        "Id": orig_test["Id"],
        "SalePrice": Y_pred
    })
submission.to_csv(submit_name, index=False)
print(submit_name)

