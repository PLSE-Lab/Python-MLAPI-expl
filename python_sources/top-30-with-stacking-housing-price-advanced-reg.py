#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge
from sklearn.svm import SVR
import category_encoders as ce
import xgboost as xgb
import lightgbm as lgbm
from mlxtend.regressor import StackingCVRegressor
import warnings

warnings.simplefilter('ignore')

# ======================================______Reading data________=====================================================
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Id = df_test['Id']

pd.set_option('display.max_columns', 90)
pd.set_option('display.max_rows', 90)


# ===================================________Data Exploration________==================================================


def data_exploration(data):
    """
    Understanding data to make better feature engineering
    :param data: Data to be explored
    :return: None
    """
    # ============______Basic FAMILIARIZATION________==================
    print('______________DATA HEAD__________ \n', data.head())
    print('______________DATA DESCRIBE______ \n', data.describe())
    print('______________DATA INFO__________ \n', data.info())

    # ===========_______DATA FREQUENT TERM___________===================
    print('_____________Total unique values in data_______ \n', data.nunique())
    print('___________________ DATA UNIQUE VALUES_____________ \n')
    print('\n', [pd.value_counts(data[cols]) for cols in data.columns], '\n')

    # ===========_______DATA CORRELATION_____________====================
    corr_matrix = data.corr(method='spearman')
    corr_matrix_salePrice = corr_matrix['SalePrice'].sort_values(ascending=False)
    corr_mat_high_df = data[corr_matrix_salePrice[corr_matrix_salePrice.values > 0.55].index]
    print('________CORRELATION MATRIX BY PRICING________ \n', corr_matrix_salePrice)
    corr_mat_graph(corr_matrix, 'Data Exploration phase')

    # =================____________DISTRIBUTION VISUALIZATION_________=================
    dist_plot(corr_mat_high_df)

    # ======================___________ Outliers__________________======================
    box_plot(corr_mat_high_df)


# ================================___________GRAPHS FUNCTIONS____________==============================================


def corr_mat_graph(cor_mat, title):
    """
    function to plot correlation matrix for better understanding of data
    :param cor_mat: correlation matrix
    :param title: Title of the graph
    :return: None
    """
    print('\n \n ____________________CORRELATION MATRIX_______________ \n \n')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cor_mat, square=False, linewidths=0.5, ax=ax, vmax=0.8, vmin=0.42)
    ax.title.set_text(title)


def dist_plot(data):
    """
    Function to plot subplots of distribution for numerical data
    :param data: data which needs to be plotted
    :return: None
    """
    print('\n \n ________________________DISTRIBUTION PLOT___________________ \n \n')
    # Plotting numerical graph
    data_filed = data.fillna(data.mean())

    for cols in data.columns:
        fig, ax = plt.subplots()
        sns.distplot(data_filed[cols])
        ax.title.set_text(cols)


def box_plot(data):
    """
    To find oultliers in the data
    :param data: data to be plot
    :return:
    """
    print('\n \n ________________________BOX PLOT___________________ \n \n')
    for cols in data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data[cols])
        ax.title.set_text(cols)


# ===========================_____________________FEATURE ENGINEERING__________________ =============================


def feature_engg(data):
    """
    Imputing missing data
    :param data: dataset to be imputed
    :return: imputed data
    """
    # =============================____________IMPUTING MISSING DATA___________=============================

    # Filling NA values with appropriate values
    discared_features = ['Alley', 'Id']
    data.drop(discared_features, axis=1, inplace=True)

    data['YearBuilt'].fillna(data['YearBuilt'].median(), inplace=True)
    data['YearRemodAdd'].fillna(data['YearRemodAdd'].median(), inplace=True)
    # Since BsmtQual, cond and exposure is NA for value where there is no basement
    data['BsmtQual'].fillna('None', inplace=True)
    data['BsmtCond'].fillna('None', inplace=True)
    data['BsmtExposure'].fillna('None', inplace=True)
    data['BsmtFinType1'].fillna('None', inplace=True)
    data['BsmtFinSF1'].fillna(0, inplace=True)
    data['BsmtFinType2'].fillna('None', inplace=True)
    data['BsmtFinSF2'].fillna(0, inplace=True)
    data['BsmtUnfSF'].fillna(0, inplace=True)
    data['TotalBsmtSF'].fillna(0, inplace=True)

    # FirePlaces And Quality
    data['Fireplaces'].fillna(0, inplace=True)
    data['FireplaceQu'].fillna('None', inplace=True)

    # Garage values also contain no garage as na hence fixing it
    data['GarageType'].fillna('None', inplace=True)
    data['GarageYrBlt'].fillna(0, inplace=True)
    data['GarageFinish'].fillna('None', inplace=True)
    data['GarageQual'].fillna('None', inplace=True)
    data['GarageCond'].fillna('None', inplace=True)

    # PoolQc filling
    data['PoolQC'].fillna('None', inplace=True)
    data['Fence'].fillna('None', inplace=True)

    # Misc features
    data['MiscFeature'].fillna('None', inplace=True)

    # Filling NA with mode or mean
    cat_feature = data.select_dtypes(include='object').copy()
    numeric_feature = data.select_dtypes(exclude='object').copy()

    cat_feature.fillna(cat_feature.mode(), inplace=True)
    numeric_feature.fillna(numeric_feature.mean(), inplace=True)

    # Merging both features
    data = pd.concat([numeric_feature, cat_feature], axis=1)

    # ====================______________________OUTLIERS_____________=================
    # Handle Outliers only in training set hence

    if 'SalePrice' in data.columns:
        # Misc Subclass Contains outliers after 130
        data = data.drop(data[data.MSSubClass > 120].index)

        # LotFrontage after 185
        data = data.drop(data[data.LotFrontage > 150].index)

        # Lot Area after 72000
        data = data.drop(data[data.LotArea > 66000].index)

        # MasVnrArea after 1180
        data = data.drop(data[data.MasVnrArea > 1100].index)

        # BsmtFinSF1 after 2300
        data = data.drop(data[data.BsmtFinSF1 > 2200].index)
        data = data.drop(data[data.BsmtFinSF2 > 1050].index)
        data = data.drop(data[data.TotalBsmtSF > 3170].index)

        # GrLivArea
        data = data.drop(data[data.GrLivArea > 4000].index)

    # =============================______________________CONVERSION OF SOME FEATURES____________=================
    data['MoSold1'] = data['MoSold'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                          7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

    # Overall condition is numerical value not very satisfying as it is out of context
    data['OverallCond'] = data['OverallCond'].apply(lambda x: 'Low' if x < 4 else ('Medium' if 4 <= x <= 6 else 'High'))

    # ============================____________________ADDING FEATURES________________________=======================
    # Adding age
    data['Age'] = 2019 - data['YearBuilt']

    # Adding total carpet area
    data['TotalCarpetArea'] = data['TotalBsmtSF'] + data['1stFlrSF']                               + data['2ndFlrSF'] + data['LowQualFinSF'] + data['GarageArea']

    # Adding season based on months Northern hemisphere
    data['season'] = data['MoSold'].apply(lambda x: 'Spring' if 3 <= x <= 5 else ('Summer' if 6 <= x <= 8
                                                                                  else ('Autumn' if 9 <= x <= 11
                                                                                        else 'Winter')))

    # Adding total bathroom
    data['TotalBathroom'] = data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']) + data['FullBath']                             + (0.5 * data['HalfBath'])

    # Adding Total sqft
    data['TotalSqF'] = data['GrLivArea'] + data['TotalBsmtSF']
    # Adding Remodeled or not
    data.loc[(data['YearBuilt'] == data['YearRemodAdd']), 'Remod'] = 'No'
    data.loc[(data['YearBuilt'] != data['YearRemodAdd']), 'Remod'] = 'Yes'

    data['HasPool'] = data['PoolArea'].apply(lambda x: 'Yes' if x > 0 else 'No')
    data['HasGarage'] = data['GarageArea'].apply(lambda x: 'Yes' if x > 0 else 'No')
    data['HasFirePlace'] = data['Fireplaces'].apply(lambda x: 'Yes' if x > 0 else 'No')
    data['Has2ndFloor'] = data['2ndFlrSF'].apply(lambda x: 'Yes' if x > 0 else 'No')

    # ==========================_______________________LOG OF NUMERICAL DATA________________=========================
    # add condition of neglecting log of salePrice if accuracy differ on competiton
    for cols in data.select_dtypes(exclude='object').columns:
            data[cols] = np.log1p(data[cols])

    # ==============================__________________DISCARD IRRELEVANT DATA____________======================
    cols_dis = ['YrSold', 'KitchenAbvGr', 'YearRemodAdd', 'BedroomAbvGr', 'FullBath', 'BsmtFullBath', 'Fireplaces',
                'HalfBath', 'BsmtHalfBath', 'MoSold', 'LotFrontage', 'MSSubClass', 'OpenPorchSF', 'GarageArea',
                'EnclosedPorch', 'WoodDeckSF', '3SsnPorch', 'ScreenPorch', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtFinSF2',
                'PoolArea', 'MiscVal', 'Street', 'Utilities', 'LandSlope', 'CentralAir', 'PavedDrive', 'LandContour',
                'Condition1', 'Condition2', 'BldgType', 'RoofMatl', 'ExterCond', 'BsmtCond', 'BsmtFinType2',
                'BsmtFinSF2', 'Heating', 'Electrical', 'LowQualFinSF', 'Functional', 'GarageQual', 'GarageCond',
                'MiscVal', ]

    data.drop(cols_dis, axis=1, inplace=True)

    # ===============================________________________DATA EXPLORE_____________==============
    if 'SalePrice' in data.columns:
        data_exploration(data)
        pass

    return data


# ===========================================___________TRAIN VALID SPLIT___________=============================
data_exploration(df_train)

x_train, x_valid = train_test_split(df_train, random_state=42)
x_train = feature_engg(x_train.copy())
x_valid = feature_engg(x_valid.copy())
df_test = feature_engg(df_test.copy())

print('_____________TRAIN TEST SPLIT_______________')
y_train = x_train['SalePrice']
x_train.drop('SalePrice', axis=1, inplace=True)

y_valid = np.expm1(x_valid['SalePrice'])
x_valid.drop('SalePrice', axis=1, inplace=True)

print('Train feature and target size', x_train.shape, y_train.shape)
print('Valid feature and target size', x_valid.shape, y_valid.shape)
print('Test feature size', df_test.shape)

# =========================_____________________CATEGORY ENCODERS______________=================
cat_features_columns = x_train.select_dtypes(include='object').columns
cat_enc = ce.OneHotEncoder()
# =============================_______________________STANDARD SCALER_______________=============================
sc = StandardScaler()

# =============================___________________PIPELINE___________________==========================
pipe = Pipeline(steps=[('cat', cat_enc), ('sc', sc)])
x_train = pipe.fit_transform(x_train)
x_valid = pipe.transform(x_valid)
x_test = pipe.transform(df_test)


# ===================================__________________________GRID SEARCH_____________________========================


def grid_search(estimator, param, X_train=x_train, Y_train=y_train, X_valid=x_valid, Y_valid=y_valid):
    print('_________________GRID SEARCH_______________________')
    grid = GridSearchCV(estimator, param, cv=3, n_jobs=-1)
    grid.fit(X_train, Y_train)
    print('Best param', grid.best_params_)
    model = grid.best_estimator_
    print('_____________________PREDICTING___________________')
    # Predicting
    predict = np.expm1(model.predict(X_valid))
    predictions = cross_val_predict(model, X_train, Y_train, cv=3)
    print('RMLSE regressor cross val', np.sqrt(mean_squared_log_error(Y_train, predictions)))
    print('RMLSE regressor valid', np.sqrt(mean_squared_log_error(Y_valid, predict)))
    return model


# ================================_____________________MODEL__________________=================================

# __________________________LASSO REG________________________
lasso_reg = LassoCV(random_state=42)
lasso_pram = {'fit_intercept': [True],
              'eps': [1e-3],
              'max_iter': [1000],
              'cv': [5],
              'tol': [0.01, 0.1]
              }
model_lasso = grid_search(lasso_reg, lasso_pram)
# RMLSE regressor valid 0.12251966389782461

# ________________________________ELASTICNETCV____________________________
elastic_reg = ElasticNetCV(random_state=42, cv=5)
elastic_param = {'fit_intercept': [True],
                 'eps': [1e-4],
                 'tol': [0.01],
                 'l1_ratio': [0.7, 0.9],
                 'n_alphas': [200],
                 'max_iter': [1000]}
model_elastic = grid_search(elastic_reg, elastic_param)

# _________________________________RIDGE REGRESSION__________________________
ridge_reg = Ridge(random_state=42)
ridge_param = {'fit_intercept': [True, False],
               'alpha': [0.1, 0.01, 1.0, 2.0],
               'normalize': [True, False],
               'tol': [0.1, 0.01]
               }
model_ridge = grid_search(ridge_reg, ridge_param)
# 0.1285584767891332



# ______________________________XGBOOST_____________________________
xgboost_reg = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_param = {'booster': ['gbtree'],
             'eta': [0.001],
             'gamma': [0.01],
             'max_depth': [6, 8],
             'lambda': [4, 6]}
model_xgb = grid_search(xgboost_reg, xgb_param)
#RMLSE regressor valid 0.13847691715756856

# _________________________SVR___________________________
svr = SVR(kernel='linear')
svr_param = {'tol': [1e-4],
             'C': [1.0, 1.1]}
model_svr = grid_search(svr, svr_param)


# ================================ _________________________STACKING ALGOS________________===================
estimators = [model_elastic, model_xgb, model_ridge, model_svr]
stregr = StackingCVRegressor(regressors=estimators, meta_regressor=model_lasso, use_features_in_secondary=True)
stregr_param = {'cv': [3, 5],
                'shuffle': [True, False]}
model_stregr = grid_search(stregr, stregr_param)

# ==================================_________________________SAVING PREDICTIONS______________==============
pred = np.expm1(model_stregr.predict(x_test))
print(pred)
# Saving result on test set
output = pd.DataFrame({'Id': Id,
                       'SalePrice': pred})

output.to_csv(r'submission.csv', index=False)
plt.show()
#RMLSE regressor valid 0.11801545186124458

