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
def load_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    y = train['SalePrice']
    y.index = train['Id']
    X = train.drop('SalePrice', axis = 1)
    X.set_index('Id', inplace = True)
    test.set_index('Id', inplace = True)
    features = list(X.columns)                  # extract all features
    quantitative = list(X.describe().columns)    # obtain quantitative features
    category = [val for val in features if val not in quantitative] # obtain categorical features
    return X, y, test, quantitative, category

def clean_data(X):
    features = list(X.columns)                  # extract all features
    quantitative = list(X.describe().columns)    # obtain quantitative features
    category = [val for val in features if val not in quantitative] # obtain categorical features
    for var in features:
        if X[var].isnull().any():
            if var in quantitative:
                median = X[var].median()
                mean = X[var].mean()
                std = X[var].std()
                if abs(median-mean)<std:
                    X[var].fillna(median, inplace = True)
                else:
                    X[var].fillna(mean, inplace = True)
            elif var in category:
                if var == 'MasVnrType':
                    X[var].fillna(X[var].mode()[0], inplace = True)
                else:
                    X[var].fillna('NA', inplace = True)
    return X

def create_poly(X, deg_dict):
    '''
    a function to create polynomials from selected features in the dataset
    deg_dict should contain column-degree pairs, for example:
        deg_dict = {'A':2, 'B':3, 'C':2}
    The dataset inputted into this function must be cleaned first
    '''
    from sklearn.preprocessing import PolynomialFeatures
    for j in deg_dict:
        pf = PolynomialFeatures(degree=deg_dict[j])
        pf.fit(X[j].values.reshape(-1,1))
        cols = [j+'pow'+str(i) for i in range(2,deg_dict[j]+1)]
        X[cols] = pd.DataFrame(pf.transform(X[j].values.reshape(-1,1)), index = X.index).iloc[:,2:]
    return X

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    return (((np.log(y+1)-np.log(y_pred+1))**2).sum()/len(y))**0.5
    
X, y, test, quant, cat = load_data()
# I only use quantitative data for training now (note: some quantitative data
# here actually categorical, but the categories are stated using numbers)
X = clean_data(X)[quant]
test = clean_data(test)[quant]
deg = {'1stFlrSF':3,'2ndFlrSF':2, 
       'FullBath':2, 'GarageArea':2,
       'GarageCars':2, 'GrLivArea':2,
       'OverallQual':2, 'TotalBsmtSF':3,
       'YearBuilt':2}
X = create_poly(X, deg)
test = create_poly(test, deg)
#X, test = scale_data(X, test)
selected_features = ['1stFlrSFpow2',
                     '1stFlrSFpow3',
                     '2ndFlrSFpow2',
                     'BedroomAbvGr',
                     'BsmtFullBath',
                     'BsmtHalfBath',
                     'Fireplaces',
                     'FullBath',
                     'FullBathpow2',
                     'GarageAreapow2',
                     'GarageCarspow2',
                     'GarageYrBlt',
                     'GrLivAreapow2',
                     'HalfBath',
                     'KitchenAbvGr',
                     'LotArea',
                     'LotFrontage',
                     'MSSubClass',
                     'MasVnrArea',
                     'MoSold',
                     'OpenPorchSF',
                     'OverallCond',
                     'OverallQual',
                     'OverallQualpow2',
                     'ScreenPorch',
                     'TotalBsmtSFpow2',
                     'TotalBsmtSFpow3',
                     'WoodDeckSF',
                     'YearBuilt',
                     'YearBuiltpow2',
                     'YearRemodAdd',
                     'YrSold']
X = X[selected_features]
test = test[selected_features]
# the training part
# importing necessary modules, I want to use Ridge and Lasso and then compare which
# one gives the best result
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# part 1: modeling the data using ridge regression
print('='*80)
print(' '*33+'Ridge Regression')
print('='*80)
params = {'alpha': [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.3, 1, 3, 10, 30, 100]}
gsRidge = GridSearchCV(estimator=Ridge(normalize=True), param_grid=params)
gsRidge.fit(X, y)
pred = pd.Series(gsRidge.predict(X), index = y.index)
errRidge = rmsle(y, pred)
print('Result:')
print('Best parameter: ',gsRidge.best_params_)
print('Best score: ',gsRidge.best_score_)
print('Root mean square logarithmic error: ', errRidge)
print('\n')
# best alpha = 0.001

# hyperparameter tuning
hyperparams = {'alpha':[0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016]}
gs = GridSearchCV(estimator=Ridge(normalize=True), param_grid=hyperparams)
gs.fit(X, y)
pred = pd.Series(gs.predict(X), index = y.index)
err = rmsle(y, pred)
print('Result:')
print('Best parameter: ',gs.best_params_)
print('Best score: ',gs.best_score_)
print('Root mean square logarithmic error: ', err)
print('\n')
#best alpha = 0.0014

# submission
ridge = Ridge(alpha = 0.0014, normalize=True)
ridge.fit(X, y)
result = pd.DataFrame(ridge.predict(test), index = test.index, columns=['SalePrice'])
result.to_csv('submission.csv')