# Housing price - practicing regression & feature engineering
# This is a port of this R notebook to python:
# https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling
# This project's goals are:
# - to help me practice organizing a complex project
# - to practice feature engineering
# - to practice modeling
# No data analysis or exploration involved. 
#
# The original poster averaged ridge, lasso, elasticnet, XGBoost's prediction to get more accurate predictions
# as XGBoost is not able to extrapolate - meaning it cannot predict values outside of the given range in the training set
# due to the nature of tree-base methods
# However, here I only use XGBoost, and it puts me at top 27%. 
# This of course can be imporved, but for now it is good enough for me.

import pandas as pd
import numpy as np
import numbers
import xgboost as xgb
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse

## Convenient functions
def fill_missing(df, cols):
    """Replace missing text with None, missing numbers with 0"""
    df = df.copy()
    for col in cols:
        if df[col].dtype in [np.number]:
            df[col] = df[col].fillna(0)
        if df[col].dtype in [np.object, np.character]:
            df[col] = df[col].fillna('None')
    return df

def mapper(x, bins):
    """Map continuous values to bins"""
    match = False
    for key in bins.keys():
        if x in bins[key]:
            out = key
            match = True
    if match == False:
        return np.nan
    return out

def adjust_test_set(dtrain, dtest):    
    """Ensure columns of train & test sets are the same"""
    dtest = dtest.copy()
    
    cols_test_miss = [x for x in dtrain.columns if x not in list(dtest.columns) + ['SalePriceLog']]
    cols_test_new = [x for x in dtest.columns if x not in list(dtrain.columns)]
    
    for col in cols_test_miss:
        dtest[col] = 0
    dtest.drop(cols_test_new, axis=1, inplace=True)
    return dtest

## Data transformation classes
class DataCleaning(BaseEstimator, TransformerMixin):
    
    def __init__(self, rm_outlier=False):
        self.rm_outlier = rm_outlier
    
    def fit(self, *args):
        return self
    
    def transform(self, df_input, *args, **kwargs):
        df = df_input.copy()
        
        # Hot fix
        df.loc[df.GarageYrBlt == 2207, 'GarageYrBlt'] = 2007

        ## Fix poll vars
        df.loc[df.Id == 2421, 'PoolQC'] = "Ex"
        df.loc[df.Id == 2504, 'PoolQC'] = "Ex"
        df.loc[df.Id ==1170, 'PoolQC'] = "Gd"

        ## Garage
        df.loc[df.Id == 2127, 'GarageFinish'] = 'Unf'
        df.loc[df.Id == 2127, 'GarageQual'] = 'TA'
        df.loc[df.Id == 2127, 'GarageCond'] = 'TA'
        
        # Build year
        check_build = df.YearBuilt <= df.YearRemodAdd
        df.loc[check_build == False , 'YearRemodAdd'] = df.loc[check_build == False, 'YearBuilt']
        
        ## Kitchen, electrical standard
        df.loc[df.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
        df.loc[df.Electrical.isnull(), 'Electrical'] = 'TA'

        # Basement        
        df.loc[df.Exterior1st.isnull(), ['Exterior1st', 'Exterior2nd']] = 'Other'
        df.loc[df.SaleType.isnull(), 'SaleType'] = 'WD'
        df.loc[df.Functional.isnull(), 'Functional'] = 'Typ'
        df.drop('Utilities', axis=1, inplace=True)
        df.loc[(df.MSSubClass == 20) & (df.MSZoning.isnull()), 'MSZoning'] = 'RL'
        df.loc[(df.MSSubClass == 30) & (df.MSZoning.isnull()), 'MSZoning'] = 'RM'
        df.loc[(df.MSSubClass == 70) & (df.MSZoning.isnull()), 'MSZoning'] = 'RM'

        # Mansory veneer
        df.loc[df.Id == 2611, 'MasVnrType'] = 'BrkCmn'
        df.loc[df.MasVnrType.isnull(), 'MasVnrType'] = 'None'
        df.loc[df.MasVnrArea.isnull(), 'MasVnrArea'] = 0

        # Lot frontage
        area_lot = df.groupby('Neighborhood')['LotFrontage'].median().reset_index(name='LotMedian')
        na_idx = df.loc[df.LotFrontage.isnull(), 'Id']
        df = df.merge(area_lot, on=['Neighborhood'], how='left')
        df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = df.loc[df.LotFrontage.isnull(), 'LotMedian']
        df.drop('LotMedian', axis=1, inplace=True)

        # Fences, Misc Feature
        df.loc[df.Fence.isnull(), 'Fence'] = 'None'
        df.loc[df.MiscFeature.isnull(), 'MiscFeature'] = 'None'

        # Fireplace
        df.loc[df.FireplaceQu.isnull(), 'FireplaceQu'] = 'None'

        # Alley
        df.loc[df.Alley.isnull(), 'Alley'] = 'None'

        # Mass filling in the rest of missing values
        df = fill_missing(df, df.columns.difference(['SalePrice']))

        # Remove outliers
        if self.rm_outlier:
            df = df.loc[df.GrLivArea <= 4000]
            df.index = range(df.shape[0])
        
        return df

class FeatureEngineering():
    
    def __init__(self):
        pass
    
    def fit(self, *args):
        return self
    
    def transform(self, df, *args, **kwargs):
               
        # Adding custom numeric features
        df_num = df.select_dtypes([int, float]).copy()
        df_char = df.select_dtypes([object]).copy()

        # Quality
        qual_cols = ['ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 
                     'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual']
        qual_list = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        for col in qual_cols:
            df_num[col + 'NUM'] = [qual_list[x] for x in df[col]]
        
        # Basement finish type
        bsmt_fin_list = {'None': 0, 'Unf': 1, 'LwQ': 2,'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        df_num['BsmtFinType1'] = [bsmt_fin_list[x] for x in df['BsmtFinType1']]
        df_num['BsmtFinType2'] = [bsmt_fin_list[x] for x in df['BsmtFinType2']]
        
        # Functionality
        functional_list = {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4,
                           'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
        df_num['Functional'] = [functional_list[x] for x in df['Functional']]
       
        # Garage finish
        garage_fin_list = {'None': 0,'Unf': 1, 'RFn': 1, 'Fin': 2}
        df_num['GarageFinish'] = [garage_fin_list[x] for x in df['GarageFinish']]
                
        # Fence types
        fence_list = {'None': 0, 'MnWw': 1, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 4}
        df_num['Fence'] = [fence_list[x] for x in df.Fence]
        
        # Dwelling type
        dwelling_list = {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 
                             70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 
                             150: 0, 160: 0, 180: 0, 190: 0}
        df_num['NewerDwelling'] = [dwelling_list[x] for x in df['MSSubClass']]
        df_num.drop('MSSubClass', axis=1, inplace=True)
        
        # Dummify var char
        df_num['RegularLotShape'] = (df['LotShape'] == 'Reg') * 1
        df_num['LandLeveled'] = (df['LandContour'] == 'Lvl') * 1
        df_num['LandSlopeGentle'] = (df['LandSlope'] == 'Gtl') * 1
        df_num['ElectricalSB'] = (df['Electrical'] == 'SBrkr') * 1
        df_num['GarageDetchd'] = (df['GarageType'] == 'Detchd') * 1
        df_num['HasPavedDrive'] = (df['PavedDrive'] == 'Y') * 1
        df_num['HasShed'] = (df['MiscFeature'] == 'Shed') * 1
        
        # convert sum number to binary        
        df_num['Remodeled'] = (df['YearBuilt'] != df['YearRemodAdd']) * 1
        df_num['RecentRemodel'] = (df['YearRemodAdd'] >= df['YrSold']) * 1
        df_num['NewHouse'] = (df['YearBuilt'] == df['YrSold']) * 1
        
        # Binary columns
        cols_binary = ['WoodDeckSF', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF', 
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        
        for col in cols_binary:
            df_num['Has' + col] = (df[col] >= 0) * 1
        df_num.drop(cols_binary, axis=1, inplace=True)
        
        df_num['HighSeason'] = (df['MoSold'].isin([5,6,7])) * 1

        # Rich
        nbrh_rich = ['Crawfor', 'Somerst, Timber', 'StoneBr', 'NoRidge', 'NridgeHt']
        df_num['NbrhRich'] = (df['Neighborhood'].isin(nbrh_rich)) * 1

        # Neighborhood
        nbrh_map = {'MeadowV': 0, 'IDOTRR': 1, 'Sawyer': 1, 'BrDale': 1, 'OldTown': 1, 'Edwards': 1, 
                     'BrkSide': 1, 'Blueste': 1, 'SWISU': 2, 'NAmes': 2, 'NPkVill': 2, 'Mitchel': 2,
                     'SawyerW': 2, 'Gilbert': 2, 'NWAmes': 2, 'Blmngtn': 2, 'CollgCr': 2, 'ClearCr': 3, 
                     'Crawfor': 3, 'Veenker': 3, 'Somerst': 3, 'Timber': 3, 'StoneBr': 4, 'NoRidge': 4, 
                     'NridgHt': 4}
        df_num['NeighborhoodBin'] = [nbrh_map[x] for x in df['Neighborhood']]

        # Heating
        heating_list = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
        df_num['HeatingScale'] = [heating_list[x] for x in df['HeatingQC']]

        # Summarize areas
        area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                     'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                     'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']

        df_num['TotalArea'] = df[area_cols].sum(axis=1)
        df_num['AreaInside'] = df['1stFlrSF'] + df['2ndFlrSF']

        # Time vars
        df_num['Age'] = 2010 - df.YearBuilt
        df_num['TimeSinceSold'] = 2010 - df.YrSold
        df_num['YearSinceRemodel'] = df.YrSold - df.YearRemodAdd
        
        # Map years to bins
        year_splits = np.array_split(sorted(range(1870, 2011)), 7)
        year_bins = {'bin{}'.format(str(counter)): year_splits[counter] for counter in range(0,7)}
        df_char['YearBuilt'] = map(lambda x: mapper(x, year_bins), df.YearBuilt)
        df_char['GarageYrBlt'] = map(lambda x: mapper(x, year_bins), df.GarageYrBlt)
        df_char['YearRemodAdd'] = map(lambda x: mapper(x, year_bins), df.YearRemodAdd)
                       
        df_num.drop(['YearBuilt', 'GarageYrBlt', 'YearRemodAdd'], axis=1, inplace=True)
        
        vars_char_drop = [
            # Quality vars
            'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 
            'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual',
            # Basement, Function, Garage, etc...
            'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 
            'Fence', 'HeatingQC',
            # Dummify
            'LotShape', 'LandContour', 'LandSlope', 'Electrical', 
            'GarageType', 'PavedDrive', 'MiscFeature', 'Neighborhood'
        ]
        
        df_char.drop(vars_char_drop, axis=1, inplace=True)
        
        # Combine
        df_output = pd.concat([df_num, df_char], axis=1)
        df_output.drop(['Id'], inplace=True, axis=1)
        
        return df_output
        
## Data preprocessing classes
class PreProcessing(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, *args):
        return self
    
    def transform(self, df_input, *args, **kwargs):
        
        df = df_input.copy()
        
        if 'SalePrice' in df_input.columns:
            sale_price_log = np.log(df['SalePrice'] + 1)
            df.drop('SalePrice', axis=1, inplace=True)

        df_num = df.select_dtypes([int, float]).copy()
        df_char = df.select_dtypes([object]).copy()

        # Convert some vars to log
        vars_skewed = df_num.columns[abs(df_num.skew()) > 0.8]
        df_num[vars_skewed] = df_num[vars_skewed].apply(lambda x: np.log(x + 1))
        df_num.rename(columns={col: col + '_log' for col in vars_skewed})

        # Normalize data
        scaler = StandardScaler().fit(df_num)
        df_num_scl = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns)

        # Get dummies
        df_dummy = pd.get_dummies(df_char, drop_first=True)
        vars_dummified = list(df_dummy.columns)

        # Combine
        df_output = pd.concat([df_num_scl, df_dummy], axis=1)
        
        if 'SalePrice' in df_input.columns:
            df_output['SalePriceLog'] = sale_price_log

        return df_output

## Data transform 
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

train_pipe = Pipeline([
    ('DataCleaning', DataCleaning(rm_outlier=True)),
    ('FeatureEngineering', FeatureEngineering()),
    ('PreProcessing', PreProcessing())
])

test_pipe = Pipeline([
    ('DataCleaning', DataCleaning(rm_outlier=False)),
    ('FeatureEngineering', FeatureEngineering()),
    ('PreProcessing', PreProcessing())
])

train = train_pipe.fit_transform(train_raw)
test = test_pipe.fit_transform(test_raw)
test = adjust_test_set(train, test)

## Modeling
X_train = train.drop('SalePriceLog', axis=1) 
y_train = np.array(train.SalePriceLog)
X_test = test

# xgboost
dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
dtest_xgb = xgb.DMatrix(X_test)

params_default = {
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.1,
    'max_delta_step': 0,
    'max_depth': 6,
    'min_child_weight': 1,
    'missing': None,
    'n_estimators': 150,
    'nthread': 4,
    'objective': 'reg:linear',
    'random_state': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'silent': False,
    'subsample': 1
}

# Tuning
xgb_grid = {
    'booster': ['gbtree'],
    'n_estimators': [750],
    'learning_rate': [0.01,0.005,0.001],
    'max_depth': [4,6,8],
    'colsample_bytree': [0.5,0.8,1],
    'min_child_weight': [2],
    'subsample': [0.2,0.4,0.6, 0.8],
    'gamma': [0.01],
}

# Uncomment this to tune it yourself
# xgbcv_result = xgb.cv(params=params_default, 
#                       dtrain=dtrain_xgb, 
#                       num_boost_round=500,                       
#                       early_stopping_rounds=50, 
#                       nfold=5, verbose_eval=50)
# params_default['n_estimators'] = xgbcv_result.shape[0]
# xgb_def = xgb.XGBRegressor(**params_default)
# xgb_gsearch = GridSearchCV(estimator=xgb_def, 
#                            param_grid=xgb_grid, 
#                            cv=5, 
#                            verbose=1)
# xgb_gsearch.fit(X_train, y_train)

best_params = {
    'booster': 'gbtree',
    'colsample_bytree': 0.5,
    'gamma': 0.01,
    'learning_rate': 0.01,
    'max_depth': 8,
    'min_child_weight': 2,
    'n_estimators': 750,
    'subsample': 0.4
}

params_tuned = params_default.copy()
for key in best_params.keys():
    params_tuned[key] = best_params[key]

xgb_tuned = xgb.XGBRegressor(**params_tuned)
xgb_tuned.fit(X_train, y_train)
y_train_xgb_tuned = xgb_tuned.predict(X_train)
y_test_xgb_tuned = xgb_tuned.predict(X_test[X_train.columns])
rmse_xgb = np.sqrt(mse(y_train, y_train_xgb_tuned))

# RidgeCV
ridge_cv = RidgeCV(alphas=(0.1, 1, 3, 5, 10), store_cv_values=True)
ridge_cv.fit(X_train, y_train)
y_train_ridge = ridge_cv.predict(X_train)
y_test_ridge = ridge_cv.predict(X_test)
rmse_ridge = np.sqrt(mse(y_train, y_train_ridge))

# LassoCV
lasso_cv = LassoCV()
lasso_cv.fit(X_train, y_train)
y_train_lasso = lasso_cv.predict(X_train)
y_test_lasso = lasso_cv.predict(X_test)
rmse_lasso = np.sqrt(mse(y_train, y_train_lasso))

# ElasticNetCV
elasnet_cv = ElasticNetCV()
elasnet_cv.fit(X_train, y_train)
y_train_elasnet = elasnet_cv.predict(X_train)
y_test_elasnet = elasnet_cv.predict(X_test)
rmse_elasnet = np.sqrt(mse(y_train, y_train_elasnet))

# Average all prediction results
y_train_avg = (y_train_ridge + y_train_lasso + y_train_elasnet + y_train_xgb_tuned) / 4
y_test_avg = (y_test_ridge + y_test_lasso + y_test_elasnet + y_test_xgb_tuned) / 4
y_test_avg_usd = (np.exp(y_test_ridge) - 1 + \
                    np.exp(y_test_lasso) - 1 + \
                    np.exp(y_test_elasnet) - 1 + \
                    np.exp(y_test_xgb_tuned) - 1) / 4
rmse_avg = np.sqrt(mse(y_train, y_train_avg))

print(rmse_xgb, rmse_ridge, rmse_lasso, rmse_elasnet, rmse_avg)

## Prepare submission file. 
# XGBoost alone is better than other models, as well ass ensemble of all models
submission = pd.DataFrame({
    'Id': range(1461, 2920),
    'SalePrice': np.exp(y_test_xgb_tuned) - 1
})

submission.to_csv('submission_xgb.csv', index=False)
