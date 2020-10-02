import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

special_columns = ['SaleCondition', 'SaleType', 'MiscFeature', 'Fence', 'PoolQC', 'Utilities', 'LandContour',
                   'LotShape', 'Alley', 'Street', 'MSZoning', 'Condition1', 'Condition2', 'PavedDrive',
                   'BldgType', 'HouseStyle', 'Neighborhood', 'LandSlope', 'LotConfig', 'RoofMatl', 'RoofStyle',
                   'GarageCond', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'GarageQual', 'ExterCond',
                   'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                   'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish',
                   'FireplaceQu']

for c in special_columns:
    train = train.drop(c, axis=1)
    test = test.drop(c, axis=1)
    

train = train.fillna(0)
test = test.fillna(0)

subject = train.drop('SalePrice', axis=1)
prices = train['SalePrice']

#train_test = train_test_split(subject, prices, test_size=0.2)

rfr = RandomForestRegressor(n_estimators=20, max_features=20)
ada = AdaBoostRegressor(base_estimator=rfr, n_estimators=40, learning_rate=0.8)

estimators = [
    ('sca', StandardScaler()),
   # ('pca', PCA()),
    ('clf', ada)
]

pipe = Pipeline(estimators)

pipe.fit(subject, prices)

preds = pipe.predict(test)

results = pd.DataFrame()

results['Id'] = test['Id']
results['SalePrice'] = preds

results.to_csv('results.csv', index=False)