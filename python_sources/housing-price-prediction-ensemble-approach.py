# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, VotingRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFECV, RFE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sc = StandardScaler()
enc = OneHotEncoder()
object_cols = []

for i in train_data.columns:
    if train_data[i].dtype == 'object':
        train_data[i] = train_data[i].astype('category').cat.codes
        object_cols.append(i)

train_data.interpolate(method='polynomial', order=9, limit_direction='both', inplace=True)

#
# for i in relevant_features:
#     if i not in not_required_features:
#         plt.hist(x=np.power(train_data[i], 0.75))
#         plt.show()
#         plt.title(i)
#         plt.boxplot(x=i, data=train_data)
#         #plt.show()


train_data['OverallQual+ExterQual'] = train_data['OverallQual'] + train_data['ExterQual']
train_data['Total_SF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] * train_data['GrLivArea']
train_data['GarageAgg'] = train_data['GarageArea'] * train_data['GarageCars']
train_data['QualityArea'] = train_data['OverallQual+ExterQual'] * train_data['Total_SF']
train_data['GarageAggQuality'] = train_data['GarageAgg'] * train_data['OverallQual+ExterQual']
train_data['SFGarageAgg'] = train_data['Total_SF'] * train_data['GarageAgg']
train_data['YearCombo'] = train_data['YearBuilt'] + train_data['YearRemodAdd']
train_data['Bath_Area'] = train_data['FullBath'] * train_data['GrLivArea']
train_data['YearBuiltQuality'] = train_data['YearCombo'] * train_data['OverallQual+ExterQual']
train_data['Lot'] = train_data['MSZoning'] * train_data['LotFrontage'] + train_data['LotArea']
train_data['QualityAreaCombo_1'] = train_data['Total_SF'] / train_data['OverallQual+ExterQual']
train_data['QualityAreaCombo_2'] = train_data['OverallQual+ExterQual'] / train_data['Total_SF']
# train_data['GarageAggQlt_1'] = train_data['GarageAgg']/train_data['GarageQual']
# train_data['GarageAggQlt_2'] = train_data['GarageQual']/train_data['GarageAgg']
train_data['Year_Area_Combo_1'] = train_data['Total_SF'] / train_data['YearCombo']
train_data['Year_Area_Combo_2'] = train_data['YearCombo'] / train_data['Total_SF']
train_data['SaleDetails'] = train_data['YearBuilt'] / train_data['MoSold']
train_data['GarageQuality'] = train_data['GarageAgg'] * train_data['GarageQual'] + train_data['GarageYrBlt']
# train_data['GarageDetails'] = train_data['GarageYrBlt'] + train_data['GarageFinish']
train_data['SellTime'] = train_data['MoSold'] * train_data['YrSold']
# train_data['PorchArea'] = train_data['Total_SF'] + train_data['OpenPorchSF']
# train_data['OverallAreaQuality'] = train_data['OverallQual+ExterQual'] * train_data['Total_SF']
# train_data['BasementDetails'] = train_data['BsmtQual'] + train_data['BsmtCond'] * train_data['Heating']
# train_data['BasementImp'] = train_data['BasementDetails'] * train_data['BsmtFinType1'] + train_data['BsmtFinType2']
# train_data['Miscelleneous'] = train_data['MiscFeature'] * train_data['MiscVal']
# train_data['Garage_Basement'] = train_data['GarageDetails'] + train_data['BasementDetails']
# train_data['SellTime_Porch'] = train_data['SellTime'] * train_data['PorchArea']
# train_data['OverallAreaQuality_YearCombo'] = train_data['OverallAreaQuality'] * train_data['YearCombo']
# train_data['Total_SF_YBQ'] = train_data['YearBuiltQuality'] * train_data['Total_SF']
# train_data['SellTime_GarageQuality'] = train_data['SellTime'] + train_data['GarageQuality']
# train_data['SellTime_Year'] = train_data['YearCombo'] + train_data['SellTime']
# train_data['Basement_Sell'] = train_data['BasementImp'] * train_data['SellTime']
# train_data['Lot_GarageDetails'] = train_data['Lot'] * train_data['GarageDetails']

train_data[['YearBuilt', 'MSSubClass', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']] = train_data[
    ['YearBuilt', 'MSSubClass', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']].astype('category')
train_data['MoSold_Sine'] = 2 * np.pi * np.sin(
    np.mean(train_data['MoSold']) - train_data['MoSold'] / (max(train_data['MoSold']) - min(train_data['MoSold'])))
train_data['MoSold_Cosine'] = 2 * np.pi * np.cos(
    np.mean(train_data['MoSold']) - train_data['MoSold'] / (max(train_data['MoSold']) - min(train_data['MoSold'])))

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 35,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 15,
    'verbose': 0
}

corr = train_data.corr()

target_val = abs(corr['SalePrice'])

relevant_features = target_val[target_val >= 0.65].index[:]

not_interesting_features = target_val[target_val < 0.65].index[:]

corr_relevant = train_data[relevant_features].corr()

not_required_features = ['ExterQual', 'GarageCars', 'SalePrice', 'Id']

z = np.abs(stats.zscore(train_data[relevant_features]))

relevant_features_df = train_data[relevant_features]

relevant_features_df = relevant_features_df[(z < 3).all(axis=1)]

train_data[relevant_features] = relevant_features_df

train_data.dropna(inplace=True)

print(train_data.shape)

label = train_data['SalePrice']

train_data.drop(columns=not_required_features, inplace=True)

fs = RFECV(estimator=DecisionTreeRegressor(max_depth=6, min_samples_split=10, random_state=10), cv=5,
           min_features_to_select=60)

td = fs.fit(train_data, label)

train_data_1 = td.transform(train_data)

train_data_1 = sc.fit_transform(train_data_1)

print(train_data_1.shape)

for i in test_data.columns:
    if test_data[i].dtype == 'object':
        test_data[i] = test_data[i].astype('category').cat.codes

test_data.interpolate(method='polynomial', order=9, limit_direction='both', inplace=True)

#
# for i in relevant_features:
#     if i not in not_required_features:
#         plt.hist(x=np.power(test_data[i], 0.75))
#         plt.show()
#         plt.title(i)
#         plt.boxplot(x=i, data=train_data)
#         plt.show()

test_data['OverallQual+ExterQual'] = test_data['OverallQual'] + test_data['ExterQual']
test_data['Total_SF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] * test_data['GrLivArea']
test_data['GarageAgg'] = test_data['GarageArea'] * test_data['GarageCars']
test_data['QualityArea'] = test_data['OverallQual+ExterQual'] * test_data['Total_SF']
test_data['GarageAggQuality'] = test_data['GarageAgg'] * test_data['OverallQual+ExterQual']
test_data['SFGarageAgg'] = test_data['Total_SF'] * test_data['GarageAgg']
test_data['YearCombo'] = test_data['YearBuilt'] + test_data['YearRemodAdd']
test_data['Bath_Area'] = test_data['FullBath'] * test_data['GrLivArea']
test_data['YearBuiltQuality'] = test_data['YearCombo'] * test_data['OverallQual+ExterQual']
test_data['Lot'] = test_data['MSZoning'] * test_data['LotFrontage'] + test_data['LotArea']
test_data['QualityAreaCombo_1'] = test_data['Total_SF'] / test_data['OverallQual+ExterQual']
test_data['QualityAreaCombo_2'] = test_data['OverallQual+ExterQual'] / test_data['Total_SF']
# test_data['GarageAggQlt_1'] = test_data['GarageAgg']/test_data['GarageQual']
# test_data['GarageAggQlt_2'] = test_data['GarageQual']/test_data['GarageAgg']
test_data['Year_Area_Combo_1'] = test_data['Total_SF'] / test_data['YearCombo']
test_data['Year_Area_Combo_2'] = test_data['YearCombo'] / test_data['Total_SF']
test_data['SaleDetails'] = test_data['YearBuilt'] / test_data['MoSold']
test_data['GarageQuality'] = test_data['GarageAgg'] * test_data['GarageQual'] + test_data['GarageYrBlt']
# test_data['GarageDetails'] = test_data['GarageYrBlt'] + test_data['GarageFinish']
test_data['SellTime'] = test_data['MoSold'] * test_data['YrSold']
# test_data['PorchArea'] = test_data['Total_SF'] + test_data['OpenPorchSF']
# test_data['OverallAreaQuality'] = test_data['OverallQual+ExterQual'] * test_data['Total_SF']
# test_data['BasementDetails'] = test_data['BsmtQual'] + test_data['BsmtCond'] * test_data['Heating']
# test_data['BasementImp'] = test_data['BasementDetails'] * test_data['BsmtFinType1'] + test_data['BsmtFinType2']
# test_data['Miscelleneous'] = test_data['MiscFeature'] * test_data['MiscVal']
# test_data['Garage_Basement'] = test_data['GarageDetails'] + test_data['BasementDetails']
# test_data['SellTime_Porch'] = test_data['SellTime'] * test_data['PorchArea']
# test_data['OverallAreaQuality_YearCombo'] = test_data['OverallAreaQuality'] * test_data['YearCombo']
# test_data['Total_SF_YBQ'] = test_data['YearBuiltQuality'] * test_data['Total_SF']
# test_data['SellTime_GarageQuality'] = test_data['SellTime'] + test_data['GarageQuality']
# test_data['SellTime_Year'] = test_data['YearCombo'] + test_data['SellTime']
# test_data['Basement_Sell'] = test_data['BasementImp'] * test_data['SellTime']
# test_data['Lot_GarageDetails'] = test_data['Lot'] * test_data['GarageDetails']

test_data[['YearBuilt', 'MSSubClass', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']] = test_data[
    ['YearBuilt', 'MSSubClass', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']] \
    .astype('category')
test_data['MoSold_Sine'] = 2 * np.pi * np.sin(
    np.mean(test_data['MoSold']) - test_data['MoSold'] / (max(test_data['MoSold']) - min(test_data['MoSold'])))
test_data['MoSold_Cosine'] = 2 * np.pi * np.cos(
    np.mean(test_data['MoSold']) - test_data['MoSold'] / (max(test_data['MoSold']) - min(test_data['MoSold'])))

Id = test_data['Id']

test_data.drop(columns=['1stFlrSF', 'GarageCars', 'Id'], inplace=True)

test_data_1 = td.transform(test_data)

test_data_1 = sc.transform(test_data_1)

X_train, X_test, Y_train, Y_test = train_test_split(train_data_1, label, test_size=0.2, random_state=2020)

# mlp = MLPRegressor(hidden_layer_sizes=(300, 150, 100), learning_rate_init=0.01, batch_size=10,
#                    learning_rate='adaptive',
#                    random_state=2020)

parameters = {'learning_rate': [0.1, 1, 0.01],
              'n_estimators': [800, 200, 400],
              'gamma': [1, 10, 0.1],
              'max_depth': [4, 10, 20]}
mlp = XGBRegressor()

mlp_search = GridSearchCV(estimator=mlp, param_grid=parameters, verbose=3, cv=5)

mlp_train = mlp_search.fit(X_train, Y_train)

mlp_prediction = mlp_search.predict(X_test)

print("Mean Squared Log Error XGB", np.power(mean_squared_log_error(Y_test, mlp_prediction), 0.5))

f = mlp_search.fit(train_data_1, label)

p = mlp_search.predict(test_data_1)

train_data['SalePrice'] = label

test_data['SalePrice'] = p

train_data_combine = train_data.append(test_data, ignore_index=True, sort=False)

test_data.drop(columns=['SalePrice'], inplace=True)

train_data_combine_label = train_data_combine['SalePrice']

train_data_11 = np.concatenate((train_data_1, test_data_1))

train_data_combine.drop(columns=['SalePrice'], inplace=True)

X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(train_data_11, train_data_combine_label, test_size=0.2,
                                                            random_state=2020)

parameters = {'learning_rate': [0.1, 0.5, 0.01],
              'n_estimators': [800, 200, 400, 1500],
              'gamma': [1, 10, 0.1],
              'max_depth': [4, 10, 20]}

xgb = XGBRegressor()

xgb_search = GridSearchCV(estimator=xgb, param_grid=parameters, verbose=3, cv=5)

mlp_com = xgb_search.fit(X_train_1, Y_train_1)

mlp_prediction_1 = xgb_search.predict(X_test_1)

print("Mean Squared Log Error XGB", np.power(mean_squared_log_error(Y_test_1, mlp_prediction_1), 0.5))

train_dataset = lgb.Dataset(X_train_1, Y_train_1)
eval_dataset = lgb.Dataset(X_test_1, Y_test_1, reference=train_dataset)

lgb_fit = lgb.train(params, train_dataset, num_boost_round=500)
lgb_predict = lgb_fit.predict(X_test_1, num_iteration=lgb_fit.best_iteration)

print("Mean Squared Log Error L-GBM", np.power(mean_squared_log_error(Y_test_1, lgb_predict), 0.5))

rf = RandomForestRegressor(n_estimators=265, max_features='auto', max_depth=10, min_impurity_decrease=1e-2,
                           random_state=2020, bootstrap=True)

rf_fit = rf.fit(X_train_1, Y_train_1)

rf_pred = rf.predict(X_test_1)

print("Mean Squared Log Error Random Forest", np.power(mean_squared_log_error(Y_test_1, rf_pred), 0.5))

gbm = GradientBoostingRegressor(n_estimators=285, max_features='auto', max_depth=5, min_impurity_decrease=1e-2,
                                random_state=2020)

gbm_fit = gbm.fit(X_train_1, Y_train_1)

gbm_pred = gbm.predict(X_test_1)

print("Mean Squared Log Error Gradient Boosting", np.power(mean_squared_log_error(Y_test_1, gbm_pred), 0.5))

ada = AdaBoostRegressor(base_estimator=GradientBoostingRegressor(n_estimators=285, max_depth=6, learning_rate=0.2,
                                                                 min_impurity_decrease=1e-4), n_estimators=270,
                        learning_rate=0.1, random_state=2020)

ada_com = ada.fit(X_train_1, Y_train_1)

ada_prediction_1 = ada.predict(X_test_1)

print("Mean Squared Log Error ADA-B", np.power(mean_squared_log_error(Y_test_1, ada_prediction_1), 0.5))

bag = BaggingRegressor(base_estimator=GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                                                min_impurity_decrease=1e-2), n_estimators=280,
                       bootstrap_features=True, random_state=2020)
bag_com = bag.fit(X_train_1, Y_train_1)

bag_prediction_1 = bag.predict(X_test_1)

print("Mean Squared Log Error Bagging", np.power(mean_squared_log_error(Y_test_1, bag_prediction_1), 0.5))

vc = VotingRegressor([('XGB', xgb), ('ADA', ada), ('BAG', bag), ('RF', rf), ('GBM', gbm)])

vc_fit = vc.fit(X_train_1, Y_train_1)

vc_pred = vc.predict(X_test_1)

print("Mean Squared Log Error Voting Classifier", np.power(mean_squared_log_error(Y_test_1, vc_pred), 0.5))

fit_model = xgb_search.fit(train_data_11, train_data_combine_label)

prediction = xgb_search.predict(test_data_1)

Submission = pd.DataFrame(columns=['Id', 'SalePrice'])

Submission['Id'] = Id
Submission['SalePrice'] = prediction

Submission.to_csv("Submission_File.csv", index=False)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session