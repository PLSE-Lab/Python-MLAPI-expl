#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, we test a method that will provide the best possible tradeoff between interpretability and prediction. We first make predictions that rely on a very transparent and interpretable model. For this part, we use Microsoft's ["explanable boosting machine"](https://github.com/interpretml/interpret). We then use the xgboost library to reduce our model's error of prediction. One way to put it is that our model goes as far as it is humanly interpretable. From there, we use a model with higher predictive performances to reduce the prediction mistakes.
# 
# In short:
# 
# $$(1)\;\;\; y_i = f(X_i) $$
# 
# Where $y_i$ is the target variable, $X_i$ the features vector and $f$ is an unknown data generating process.
# 
# $$(2)\;\;\; y_i = \hat{y}_i + \lambda_i $$
# 
# Where $\hat{y}_i$ is estimated with a glassbox model, $\lambda_i$ the prediction's residual.
# 
# $$(3)\;\;\; y_i = \hat{y}_i + \hat{\lambda}_i + \sigma_i $$
# 
# Where $\hat{\lambda}_i$ is estimated with a blackbox model, and $\sigma_i$ is the new residual. We hypothesize that $\sum_{i=1}^{N}\lambda_i^2 > \sum_{i=1}^{N}\sigma_i^2$.
# 
# We believe it is a better method than stacking an interpretable and blackbox models, or using ex-post sensitivity tests (like SHAP), since the additive structure sets clear boundaries between the interpretable and non-intepretable parts of each prediction. 
# 
# We use the dataset on houses sale in Ames Iowa prepared by Dean De Cock (2011). It is a widely used and some high quality publicly available notebooks already did a good job at exploring it. We can thus save some time building upon them to test our method.
# # Environment & loading data

# In[ ]:


import os
os.chdir("/kaggle/input/house-prices-advanced-regression-techniques/")  

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import uniform, randint, norm
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder, scale, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from math import sqrt      

# Installing Microsoft's package for "explanable boosting machine"
get_ipython().system('pip install -U interpret')

# Set random seed
np.random.seed(123)

# loading data
data = pd.read_csv("train.csv")


# # Preprocessing
# Some features of this dataset are notoriously problematic. In a real situation, we would investigate the nature of each variable, especially for the reasons behind the missing data. We would also investigate about outliers. Since we do not wish to spend too much time on this dataset, we rely on its author's remarks and the previous works in the data science community.
# ## (Almost) perfect correlation between features
# We drop GarageArea, since it is almost perfectly correlated with GarageCars. Same for 1stFloorSF, TotalBsmtSF and GrLivArea, TotRmsAbvGrd. We keep the later in both cases.
# ## Intended missing values
# Most NAs are tagged so voluntarily. For example, the data dictionnary indicates that the PoolQc variable is missing if the property has no pool. We will thus replace them by the string "no", which will not be interpreted as missing.
# ## Other missing values
# Looking at the remaining missing values, we find that LotFrontage, that is the "Linear feet of street connected to property" has more than 15% of NAs. For now, we do not have an explanation for this. We will thus simply remove this feature. We do the same for the variable GarageYrBlt.
# 
# The three remaining features have less than one percent of NAs. We will deal with them in the preprocessing pipeline. The two numeric NAs will be changed for the median of the respective variable and the NA for the variable Electrical will take is most frequent value.
# ## Outliers
# There are five points that the author of the dataset identified as outliers. Three of them are partial sales that simply are another kind of transaction. We thus follow the recommendation of the author by removing all transactions with more than 4000 square feets of total living area (above ground). There are simply not such enough cases in the dataset to properly train a model.

# In[ ]:


# droping (almost) perfectly correlated variables
data.drop(["GarageArea", "1stFlrSF", "GrLivArea"], axis=1)

# replacing intended NAs
NA_to_no = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
            "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", 
            "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for i in NA_to_no:
  data[i] = data[i].fillna("N")

# Droping the two features with many missing values
data = data.drop(["LotFrontage", "GarageYrBlt"], axis = 1)

#Dropping the outliers
data = data[data.GrLivArea<4000]

# Splitting the features from the target, and the train and test sets

X = data
X = X.drop("SalePrice", axis=1)
y = data.loc[:,"SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

# identifying the categorical and numeric variables

categorical = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig","LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MoSold", "SaleType", "SaleCondition", "BedroomAbvGr", "KitchenAbvGr"]

numeric = ["LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "2ndFlrSF", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "YrSold"]


# # Linear model

# In[ ]:


# I use the log transformation for prediction

def log(x):
    return np.log(x)

def exp(x):
    return np.exp(x)

# Setting up the preprocessor.
preprocessor = make_column_transformer((make_pipeline(SimpleImputer(strategy="most_frequent"), 
                                                      OneHotEncoder(handle_unknown="ignore")), categorical), 
                                       (make_pipeline(SimpleImputer(strategy="median"),
                                                      StandardScaler()), numeric))

# Instantiating the model
pipeline_linear = make_pipeline(preprocessor,
                               TransformedTargetRegressor(LinearRegression(),
                               func=log, inverse_func =exp))

#Fitting the model and retrieving the prediction
pipeline_linear.fit(X_train, y_train)
line_pred = pipeline_linear.predict(X_test)


# # xgboost

# In[ ]:


pipeline_xgb = make_pipeline(preprocessor,
                    TransformedTargetRegressor(xgb.XGBRegressor(objective ='reg:squarederror', nthread=-1), 
                                               func=log, inverse_func=exp))
# Hyperparameters distributions
params = {
    "transformedtargetregressor__regressor__colsample_bytree": uniform(0.7, 0.3),
    "transformedtargetregressor__regressor__gamma": uniform(0, 0.5),
    "transformedtargetregressor__regressor__learning_rate": uniform(0.03, 0.3),
    "transformedtargetregressor__regressor__max_depth": randint(2, 6),
    "transformedtargetregressor__regressor__n_estimators": randint(500, 1000),
    "transformedtargetregressor__regressor__subsample": uniform(0.6, 0.4)
}

# Instantiating the xgboost model, with random-hyperparameter tuning
xgb_model = RandomizedSearchCV(pipeline_xgb, param_distributions=params, random_state=123, 
                               n_iter=50, cv=5, n_jobs=-1)

#Fitting the model and retrieving the predictions
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)


# # ebm

# In[ ]:


from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
from interpret.data import Marginal

# Definition of the EBM preprocessor; I do not one hot encode, since EBM deals with categoricals

preprocessor_ebm = make_column_transformer(
    (SimpleImputer(strategy="most_frequent"), categorical),
    (SimpleImputer(strategy="median"), numeric)
    )

# Instantiating the model
ebm = make_pipeline(preprocessor_ebm, 
                    TransformedTargetRegressor(ExplainableBoostingRegressor(random_state=123),
                    func=log, inverse_func=exp))

#Fitting the model and retrieving the predictions
ebm.fit(X_train, y_train)
ebm_pred = ebm.predict(X_test)


# # ebm + xgboost 

# In[ ]:


params = {
    "xgbregressor__colsample_bytree": uniform(0.7, 0.3),
    "xgbregressor__gamma": uniform(0, 0.5),
    "xgbregressor__learning_rate": uniform(0.03, 0.3),
    "xgbregressor__max_depth": randint(2, 6),
    "xgbregressor__n_estimators": randint(500, 1000),
    "xgbregressor__subsample": uniform(0.6, 0.4)
}

pipeline_xgb2 = make_pipeline(preprocessor,
                              xgb.XGBRegressor(objective ='reg:squarederror', nthread=-1))

xgb_model_2 = RandomizedSearchCV(pipeline_xgb2, param_distributions=params, random_state=123,
                                 n_iter=50, cv=5)

# getting residual predictions from the train data
ebm_pred_train = ebm.predict(X_train)
ebm_residual_train = y_train - ebm_pred_train

# training the xgb from the train data residual
xgb_model_2.fit(X_train, ebm_residual_train)
residual_predicted = xgb_model_2.predict(X_test)

# then we get our boosted ebm prediction
ebm_xgb_pred = ebm_pred + residual_predicted


# # Comparing performances
# It has been remarked in the past that ebm gives similar prediction performances than xgboost. Our method reaches performances that are in between the two.

# In[ ]:


# Getting performance 

predict = [line_pred, xgb_pred, ebm_pred, ebm_xgb_pred]

mae = []
mse = []
rmse = []

for i in predict:
    mae.append(mean_absolute_error(y_test, i))
    mse.append(mean_squared_error(y_test, i))
    rmse.append(sqrt(mean_squared_error(y_test, i)))

scores = pd.DataFrame([mae, mse, rmse], 
                      columns=["line", "xgb", "ebm", "ebm + xgb"],
                      index = ["mae", "mse", "rmse"])

scores["ebm + xgb over ebm"] = (round((scores["ebm"]/scores["ebm + xgb"] -1)*100, 2)                                .astype(str) +" %")
scores["xgb over ebm + xgb"] = (round((1- scores["xgb"]/scores["ebm + xgb"])*100, 2)                                .astype(str) +" %")

scores

