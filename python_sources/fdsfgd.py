# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('../input/train.csv')
train = pd.get_dummies(train)
test = pd.read_csv('../input/test.csv')
test = pd.get_dummies(test)
print (test)
#X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice',axis=1), train['SalePrice'], test_size=0.2)
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)
    test[feature] = np.log1p(test[feature].values)

def quadratic(feature):
    train[feature+'2'] = train[feature]**2
    test[feature+'2'] = test[feature]**2
    
log_transform('GrLivArea')
log_transform('1stFlrSF')
log_transform('2ndFlrSF')
log_transform('TotalBsmtSF')
log_transform('LotArea')
log_transform('LotFrontage')
log_transform('KitchenAbvGr')
log_transform('GarageArea')

quadratic('OverallQual')
quadratic('YearBuilt')
quadratic('YearRemodAdd')
quadratic('TotalBsmtSF')
quadratic('2ndFlrSF')
#quadratic('Neighborhood')
#quadratic('RoofMatl')
quadratic('GrLivArea')

qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
        '2ndFlrSF2', 'GrLivArea2']

boolean = ['TotalBsmtSF', 'GarageArea', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF',
            'OpenPorchSF', 'PoolArea', 'YearBuilt']
quantitative=['GrLivArea','1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','LotFrontage','KitchenAbvGr','GarageArea']

X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice',axis=1), train['SalePrice'], test_size=0.2)
features = quantitative + boolean + qdr
lasso = GradientBoostingRegressor()
X = train[features].fillna(0.).values
Y = train['SalePrice'].values
lasso.fit(X, np.log(Y))

#Ypred = np.exp(lasso.predict(X_test[features].fillna(0.).values))
#error(y_test.values,Ypred)
Ypred = np.exp(lasso.predict(test[features].fillna(0.).values))
ids=test['Id']
sub = pd.DataFrame({
        "Id": ids,
        "SalePrice": Ypred
    })
sub.to_csv("prices_submission.csv", index=False)
