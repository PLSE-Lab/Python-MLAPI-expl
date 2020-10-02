# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# print(train.dtypes)

# print(train['SaleCondition'].unique())

# # scatter to see the relationship between "GrLivArea" and "SalePrice"
# ax = sns.regplot(x="GrLivArea", y="SalePrice", data=train)
# plt.show()

# string label to categorical values
from sklearn.preprocessing import LabelEncoder

for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

# print(train['SaleCondition'].unique())

# # search for missing data
# import missingno as msno
# msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))

# # Which columns have nan?
# print('training data: ')
# for i in np.arange(train.shape[1]):
#     n = train.iloc[:,i].isnull().sum()
#     if n > 0:
#         print(list(train.columns.values)[i] + ': ' + str(n) + ' nans')
#
# print('testing data: ')
# for i in np.arange(test.shape[1]):
#     n = test.iloc[:,i].isnull().sum()
#     if n > 0:
#         print(list(test.columns.values)[i] + ': ' + str(n) + ' nans')
#

# keep ID for submission
train_ID = train['Id']
test_ID = test['Id']

# split data for training
y_train = train['SalePrice']
X_train = train.drop(['Id','SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

# dealing with missing data
Xmat = pd.concat([X_train, X_test])
Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)
Xmat = Xmat.fillna(Xmat.median())

# # check whether there are still nan
# import missingno as msno
# msno.matrix(df=Xmat, figsize=(20,14), color=(0.5,0,0))
#
# print(Xmat.columns.values)
# print(str(Xmat.shape[1]) + ' columns')

# add a new feature 'total sqfootage'
Xmat['TotalSF'] = Xmat['TotalBsmtSF'] + Xmat['1stFlrSF'] + Xmat['2ndFlrSF']
# print(Xmat.shape[1])

# # normality check for the target
# ax = sns.distplot(y_train)
# plt.show()
#
# log-transform the dependent variable for normality
y_train = np.log1p(y_train)
#
# ax = sns.distplot(y_train)
# plt.show()

# train and test
X_train = Xmat.iloc[:train.shape[0],:]
X_test = Xmat.iloc[train.shape[0]:,:]

# # Compute the correlation matrix
# corr = X_train.corr()
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
#
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# plt.show()


# feature importance using random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')

ranking = np.argsort(-rf.feature_importances_)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
# ax.set_xlabel("feature importance")
# plt.show()

# use the top 29 features only
X_train = X_train.iloc[:,ranking[:29]]
X_test = X_test.iloc[:,ranking[:29]]

# interaction between the top 2
X_train["Interaction"] = X_train["TotalSF"]*X_train["OverallQual"]
X_test["Interaction"] = X_test["TotalSF"]*X_test["OverallQual"]

# zscoring
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

# # heatmap
# f, ax = plt.subplots(figsize=(20, 9))
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
# sns.heatmap(X_train, cmap=cmap)
# plt.show()

# remove outliers
y_train = y_train.drop(y_train[(X_train['LotArea']>10)].index)
X_train = X_train.drop(X_train[(X_train['LotArea']>10)].index)

# # heatmap
# f, ax = plt.subplots(figsize=(20, 9))
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
# sns.heatmap(X_train, cmap=cmap)
# plt.show()

# # relation to the target
# fig = plt.figure(figsize=(12,7))
# for i in np.arange(6):
#     ax = fig.add_subplot(2,3,i+1)
#     sns.regplot(x=X_train.iloc[:,i], y=y_train)
#
# plt.show()

# outlier deletion
Xmat = X_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['Interaction']>6)].index)

# recover
y_train = Xmat['SalePrice']
X_train = Xmat.drop(['SalePrice'], axis=1)

# pandas to numpy
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

# linear regression and ensembling
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV

# look for the best regularization term
def regr_cvscore(regr):
    return np.mean(np.sqrt(-cross_val_score(regr, X_train, y_train, scoring='neg_mean_squared_error', cv=5)))

regrs = []
score = []
weight = np.array([0.35,0.15,0.1])

# Ridge
regr_ridge = RidgeCV(alphas=(0.1,0.11,0.12,0.13,0.14), cv=5, scoring='neg_mean_squared_error')
regr_ridge.fit(X_train, y_train)
regrs.append(regr_ridge)
score.append(regr_cvscore(regr_ridge))

# Elastic Net
regr_EN = ElasticNetCV(cv=5, random_state=None)
regr_EN.fit(X_train, y_train)
regrs.append(regr_EN)
score.append(regr_cvscore(regr_EN))

# Lasso
regr_lasso = LassoCV(cv=5, random_state=None, max_iter=10000)
regr_lasso.fit(X_train, y_train)
regrs.append(regr_lasso)
score.append(regr_cvscore(regr_lasso))

# sort performance and weighted avarage
rank = np.argsort(score)
y_pred_reg = np.zeros(len(test_ID))
for i in np.arange(3):
    print('++++++++++++++++++++++++++++++')
    print('model: ' + str(regrs[rank[i]]))
    print(str(score[rank[i]]))

    temp = regrs[rank[i]].predict(X_test)
    y_pred_reg += np.expm1(temp) * weight[i]

# XGBoost & LGBoost
import xgboost as xgb
import lightgbm as lgb

# the same parameters as
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

mdl_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

mdl_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

mdl_xgb.fit(X_train, y_train)
y_pred_xgb = np.expm1(mdl_xgb.predict(X_test))
mdl_lgb.fit(X_train, y_train)
y_pred_lgb = np.expm1(mdl_lgb.predict(X_test))

# weighted averaging models
y_pred = y_pred_reg + 0.2*y_pred_xgb + 0.2*y_pred_lgb

# for submission
submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": y_pred
    })
submission.to_csv('houseprice.csv', index=False)
