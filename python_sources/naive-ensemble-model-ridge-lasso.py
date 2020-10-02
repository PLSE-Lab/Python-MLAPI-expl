# model.py
# regression on housing prices.
# model is boosted with xgboost

# Team Name: The Oracle (Ronny Macmaster and Javier Zepeda)
# Kaggle RMSE: 12.203
# Rank : 514
 
# 2a) Pure Ridge Regression:
# (alpha = 7, features = 265, outliers > dim=12)
# Kaggle RMSE: 12.203

# 2b) Pure Lasso Regression:
# (alpha = 0.002, features = 265, outliers > dim=12)
# Kaggle RMSE: 13.614 

# 4) Ensemble Learning (Ridge + Lasso):
# (alpha = 60, features = 265, outliers > dim=12)
# Kaggle RMSE: 12.910 

# 5) Pure XGB Regression:
# (n_estimators = 380, max_depth = 2, features = 265, outliers > dim=12)
# Kaggle RMSE: 13.716 

# 6) Best Attempt (Pure Ridge Regression):
# (alpha = 7, features = 265, outliers > dim=12)
# Team Name: The Oracle (Ronny Macmaster and Javier Zepeda)
# Kaggle RMSE: 12.203
# Rank : 514

# 7/8) Helpful Kernels:
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# https://www.kaggle.com/apapiu/regularized-linear-models
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn
# https://www.kaggle.com/jimthompson/ensemble-model-stacked-model-example

# 9) Analysis of Results:
# Overall, we found the competition quite challenging. 
# Although we tried out many other models, (Ridge, Lasso, Tree, XGB Regressor, Ensemble of three)
# The model that generalized best in the end was a simple Ridge Regression model. 

# Approaches we utilized:

# Missing and NaN Data Replacement:
# We tried replacing the missing data with the mean / median, and that barely boosted our score.
# Maybe the effects would be more significant if we replaced with all 0s 
# or individually examined each NaN for an appropriate substitute.

# Feature Selection:
# At first, we tried picking a handful of numerical and categorical features by hand.
# These models performed poorly and were severly underfit (numFeatures <= 30)
# Thus, we settled for a middle ground where we keep only the top k features
# Features are ranked by their linear correlation with the SalePrice
# This approach yielded some modest improvements to the score, 
# but we ended up fitting to a suprisingly high number of features (265)

# Log Transforms and Unskewing Features:
# The SalePrice label is not normally distributed, so it needed to be log transformed.
# This was helpful for the regression models because regression expects a normally distributed label.
# It was also helpful to log transform certain highly skewed features.
# The benefits were undermined however, because many skewed parameters were left as bimodal distributions
# Small concentrations of points around 0 distorted the evident SalePrice linear correlation for nonzero values.

# Feature Normalization:
# We theorized the model might perform better if the features were fed in as Z-scores rather than values.
# Unfortunately, the model performed worse (distortion from outliers?)
# This approach was abandoned because log transforming only the skewed features was a better alternative.

# Different Linear and Nonlinear Models:
# We thought adding an XGB model or a nonlinear TreeRegressor would perform better than a linear regressor.
# Our assumption was correct, however it's because these models severly overfit the data.
# We believe it's because they form decision trees and possible handle the categorical data better.
# These models would generalize poorly on their own, but they could be combined with linear models for balance.

# Cross Validation:
# Optimizing hyperparameters proved easy when using Cross Validation. 
# It also served well as a measure of how well the model may generalize,
# However, as we journeyed into ensemble learning, it did not always serve as an accurate measure anymore. 
# It is a useful tool for determining how well a simple model may generalize / how much it overfits the training set.

# Authors:
#   Ronald Macmaster (eid: rpm953)
#   Javier Zepeda (eid: jtz236)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats import skew

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd
import numpy as np
import math

def drop_missing(data, alpha):
    # drop columns with >= alpha missing
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (total / (data.isnull().count())).sort_values(ascending=False)
    missing = pd.concat([total, percent], axis=1, keys=["total", "percent"])
    missing = missing.sort_values("percent", ascending=False)
    return data.drop(missing[missing["percent"] >= alpha].index, 1) 

# drop a certain number of weakly correlated features
def feature_selection(X, y, keep):
    qcorr = X.corrwith(y).sort_values()
    qcorr = qcorr.abs().sort_values()
    drop = len(qcorr) - keep
    return X.drop(qcorr.nsmallest(drop).index, axis=1)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X):
        return self
    def transform(self, X):
        return X[self.feature_names]

class DataFrameSubset(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X):
        return self
    def transform(self, X):
        return X.drop(X.columns.difference(self.feature_names), axis=1)

class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self
    def transform(self, X):
        return pd.get_dummies(X)

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        self.method = method
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.method == "mean":
            return X.fillna(X.mean())
        elif self.method == "median":
            return X.fillna(X.median())
        else: # default
            return X.fillna(X.mean())

class CustomNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return ((X - self.mean) / self.std)

class CustomUnskewer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_ratio):
        self.skew_ratio = skew_ratio 
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # num_features = X.select_dtypes(exclude=["object"]).columns
        skewed_feats = X.apply(lambda x: skew(x.dropna())) #compute skewness
        skewed_feats = skewed_feats[skewed_feats > self.skew_ratio].index
        X[skewed_feats] = np.log1p(X[skewed_feats])
        return X

class MissingSelector(BaseEstimator, TransformerMixin):
    def __init__(self, ratio):
        self.ratio = ratio
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return drop_missing(X, self.ratio)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, label, num_features):
        self.num_features = num_features
        self.label = label
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return feature_selection(X, self.label, self.num_features)

class OutlierDetection(BaseEstimator):
    def __init__(self, alpha, dims):
        self.alpha = alpha
        self.dims = dims
    def fit(self, X):
        std, mean, median = X.std(), X.mean(), X.median()
        X["outliers"] = 0
        for col in X.columns:
            if not col == "outliers":
                outlier_idx = (abs(X[col]) > (self.alpha * std[col] + mean[col]))
                X.set_value(outlier_idx, "outliers", X[outlier_idx]["outliers"] + 1)
        outliers = X[X["outliers"] > self.dims]
        X.drop("outliers", axis=1, inplace=True)
        return outliers.index 

# class OutlierRemoval(BaseEstimator, TransformerMixin):
#     def __init__(self, alpha, method):
#         self.alpha = alpha
#         self.method = method
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         std = X.std()
#         mean = X.mean()
#         median = X.median()
#         for col in X.columns:
#             outlier_idx = (abs(X[col]) > (self.alpha * std[col] + mean[col]))
#             X.set_value(outlier_idx, col, 0)
#         return X 

# read data
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# prep train data
label_col = "SalePrice"
index_col = "Id"
xtest = test.drop([index_col], axis=1)
xtrain = train.drop([index_col, label_col], axis=1)
y = np.log1p(train[label_col])

# all data, combo dataframe
housing = pd.concat([xtrain, xtest])

# Optional: Feature Engineering
# Transform "Numerical" categorical features
housing["MSSubClass"] = housing["MSSubClass"].apply(str)
housing["OverallCond"] = housing["OverallCond"].apply(str)
housing["YrSold"] = housing["YrSold"].apply(str)
housing["MoSold"] = housing["MoSold"].apply(str)

# Add an important feature (Total Area)
housing["TotalSF"] = housing["TotalBsmtSF"] + housing["1stFlrSF"] + housing["2ndFlrSF"]

# aggregate pipeline, preprocessing (all data)
missing_ratio = 0.05
skew_ratio = 0.25
pipeline = Pipeline([
    ("custom_binarizer", CustomBinarizer()),
    ("missing_selector", MissingSelector(missing_ratio)),
    ("custom_unskewer", CustomUnskewer(skew_ratio)),
])
samples = xtrain.shape[0] # number of training samples
housing = pipeline.fit_transform(housing)
xtrain = housing[:samples]
xtest = housing[samples:]

# training pipeline, preprocessing
num_features = 265
mean, std = xtrain.mean(), xtrain.std()
pipeline = Pipeline([
    ("selector", DataFrameSelector(xtrain.columns)),
    ("custom_imputer", CustomImputer("mean")),
    # ("custom_normalizer", CustomNormalizer(mean, std)),
    ("feature_selection", FeatureSelector(y, num_features)),
])
xtrain = pipeline.fit_transform(xtrain)

# Optional: outlier removal
outlier_removal = OutlierDetection(alpha=3, dims=12)
outlier_index = outlier_removal.fit(xtrain)
samples = xtrain.shape[0] - len(outlier_index)
xtrain = xtrain.drop(outlier_index).reset_index(drop=True)
y = y.drop(outlier_index).reset_index(drop=True)

# testing pipeline, preprocessing
pipeline = Pipeline([
    ("selector", DataFrameSelector(xtrain.columns)),
    ("custom_imputer", CustomImputer("mean")),
    # ("custom_normalizer", CustomNormalizer(mean, std)),
    
])
xtest = pipeline.fit_transform(xtest)

print ("numFeatures: %d, missingRatio: %f, skewRatio: %f" \
        % (num_features, missing_ratio, skew_ratio))

# # Visualize changes from skewing the features.
# quant_features = xtrain.select_dtypes(exclude=["object"]).columns
# # qual_features = housing.select_dtypes(include=["object"])
# for feat in quant_features:
#     qcorr = pd.DataFrame()
#     qcorr["skewed_feat"] = xtrain[feat]
#     qcorr["orig_feat"] = train[feat]
#     qcorr["SalePrice"] = y
#     plt.figure(); sns.regplot(x="skewed_feat", y="SalePrice", data=qcorr)
#     plt.figure(); sns.regplot(x="orig_feat", y="SalePrice", data=qcorr)
#     plt.title(feat)
#     plt.show()

# train some regression models
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# fit a ridge regularization model
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=7.0).fit(xtrain, y)
ridge_pred = ridge_model.predict(xtrain)
ridge_scores = cross_val_score(ridge_model, xtrain, y, scoring="neg_mean_squared_error", cv=20)
print "ridge rmse(mean, std): (%lf, %lf)" % (np.sqrt(-ridge_scores.mean()), np.sqrt(ridge_scores.std()))
print "ridge training rmse: ", np.sqrt(mean_squared_error(y, ridge_pred))

# fit a lasso regularization model
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.002).fit(xtrain, y)
lasso_pred = lasso_model.predict(xtrain)
lasso_scores = cross_val_score(lasso_model, xtrain, y, scoring="neg_mean_squared_error", cv=20)
print "lasso rmse(mean, std): (%lf, %lf)" % (np.sqrt(-lasso_scores.mean()), np.sqrt(lasso_scores.std()))
print "lasso training rmse: ", np.sqrt(mean_squared_error(y, lasso_pred))

# fit a nonlinear tree model
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor().fit(xtrain, y)
tree_pred = tree_model.predict(xtrain)
tree_scores = cross_val_score(tree_model, xtrain, y, scoring="neg_mean_squared_error", cv=10)
print "tree rmse(mean, std): (%lf, %lf)" % (np.sqrt(-tree_scores.mean()), np.sqrt(tree_scores.std()))
print "tree training rmse: ", np.sqrt(mean_squared_error(y, tree_pred))

# # Plot L0 Norm vs alpha
# norm = []
# alphas = [0.01, 0.1, 1, 5, 10, 20, 40, 80]
# for a in alphas:
#     coef = Lasso(alpha=a).fit(xtrain, y).coef_
#     norm.append(np.linalg.norm(coef, ord=0))
# plt.plot(pd.Series(norm, index=alphas))
# plt.title("L0 Norm of Lasso Coefficients")
# plt.ylabel("L0 Norm"); plt.xlabel("Alpha")
# plt.show()

# grid search a regression model (optimizes alpha)
from sklearn.model_selection import GridSearchCV
ridge_grid = {"alpha" : np.arange(0.9, 1.0, 0.02)}
lasso_grid = {"alpha" : np.arange(0.1, 0.01, -0.01)}

grid_search = GridSearchCV(lasso_model, lasso_grid, cv=10, scoring="neg_mean_squared_error")
grid_search.fit(xtrain, y)
print "best lasso alpha: ", grid_search.best_params_

grid_search = GridSearchCV(ridge_model, ridge_grid, cv=10, scoring="neg_mean_squared_error")
grid_search.fit(xtrain, y)
print "best ridge alpha: ", grid_search.best_params_

# train an xgboost model to fine tune model
import xgboost as xgb 
xgb_model = xgb.XGBRegressor(n_estimators=380, max_depth=2, learning_rate=0.1)
xgb_model = xgb_model.fit(xtrain, y)
xgb_pred = xgb_model.predict(xtrain)
xgb_scores = cross_val_score(xgb_model, xtrain, y, scoring="neg_mean_squared_error", cv=10)
print "xgb rmse(mean, std): (%lf, %lf)" % (np.sqrt(-xgb_scores.mean()), np.sqrt(xgb_scores.std()))
print "xgb training rmse: ", np.sqrt(mean_squared_error(y, xgb_pred))
joblib.dump(xgb_model, "xgb_model.pkl")

# # grid search the xgb model (optmizes model hyperparameters)
# from sklearn.model_selection import GridSearchCV
# xmodel = xgb.XGBRegressor(n_estimators=380, max_depth=2, learning_rate=0.1)
# xgb_grid = {
#     "n_estimators" : [375, 380, 385],
#     # "max_depth" : [2, 3, 4],
# }
# 
# grid_search = GridSearchCV(xmodel, xgb_grid, cv=5, scoring="neg_mean_squared_error")
# grid_search.fit(xtrain, y)
# print "best xgb grid params: ", grid_search.best_params_

# Training Predictions
xgb_model = joblib.load("xgb_model.pkl") 
pred = pd.DataFrame({
    "lasso_pred" : lasso_model.predict(xtrain), 
    "ridge_pred" : ridge_model.predict(xtrain), 
    "xgb_pred" : xgb_model.predict(xtrain),
})

# Ensemble Learning 
# Add the training predictions as features
# Then train a ridge regression model over ALL the features.
xtrain = pd.concat([xtrain, pred], axis=1) # pd.concat([xtrain, pred])
ensemble_model = Ridge(alpha = 60)
ensemble_model.fit(xtrain, y)
ensemble_pred = ensemble_model.predict(xtrain)
ensemble_scores = cross_val_score(ensemble_model, xtrain, y, scoring="neg_mean_squared_error", cv=10)
print "ensemble rmse(mean, std): (%lf, %lf)" % (np.sqrt(-ensemble_scores.mean()), np.sqrt(ensemble_scores.std()))
print "ensemble training rmse: ", np.sqrt(mean_squared_error(y, ensemble_pred))
print "ensemble model coefficients: ", ensemble_model.coef_[-len(pred.columns):]


# grid search a regression model (optimizes alpha)
from sklearn.model_selection import GridSearchCV
grid_model = ensemble_model
grid = {"alpha" : np.arange(10, 1.0, -1.00)}
grid_search = GridSearchCV(grid_model, grid, cv=10, scoring="neg_mean_squared_error")
grid_search.fit(xtrain, y)
print "best grid params: ", grid_search.best_params_


pred["label"] = y
pred["ensemble_pred"] = ensemble_pred
pred["residual"] = pred["ensemble_pred"] - pred["label"]

# prediction and residual plotting
pred.plot(x="ensemble_pred", y="residual", kind="scatter"); plt.title("Residual Plot"); 
pred.plot(x="lasso_pred", y="label", kind="scatter"); plt.title("Lasso Prediction"); 
pred.plot(x="ridge_pred", y="label", kind="scatter"); plt.title("Ridge Prediction"); 
pred.plot(x="xgb_pred", y="label", kind="scatter"); plt.title("XGB Prediction"); 
pred.plot(x="ensemble_pred", y="label", kind="scatter"); plt.title("Ensemble Prediction"); 
plt.show()

# build kaggle submission
pred = pd.DataFrame({
    "lasso_pred" : lasso_model.predict(xtest), 
    "ridge_pred" : ridge_model.predict(xtest), 
    "xgb_pred" : xgb_model.predict(xtest),
})

xtest = pd.concat([xtest, pred], axis=1)
price = ensemble_model.predict(xtest) 
results = pd.DataFrame({
    "Id" : test["Id"],
    "SalePrice" : np.expm1(price),
})

# print results["SalePrice"]
results.to_csv("results.csv", index=False)