
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

import sklearn
from sklearn import preprocessing

data = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# We remove 'MSZoning', 'Functional', 'SaleType' 'KitchenQual' 'Utilities'
# 'Condition2' 'RoofMatl' 'MiscFeature' 'Heating' 'GarageQual' 'PoolQC'
# 'Exterior1st', 'Exterior2nd', 'Electrical' 'HouseStyle'
# because the test data contains entries that are not found in the 
# training data.

categorical_features = ['Street','Alley', 'LotShape',
                        'LandContour', 'LotConfig',
                        'LandSlope','Condition1',
                        'BldgType','RoofStyle',
                        'ExterQual', 'ExterCond','Foundation',
                        'BsmtQual','BsmtCond','BsmtExposure', 
                        'BsmtFinType1','BsmtFinType2',
                        'HeatingQC','CentralAir',
                        'FireplaceQu',
                        'GarageType','GarageFinish',
                        'GarageCond','PavedDrive','Fence',
                        'SaleCondition',
                        'Neighborhood','MasVnrType']
                        
numerical_features = ['LotFrontage', 'FullBath',
                        'OpenPorchSF', '3SsnPorch', 'OverallQual', 
                        'BedroomAbvGr', 'TotRmsAbvGrd', 
                        'MSSubClass', 'YrSold', 
                        'YearRemodAdd', 'HalfBath', 'PoolArea', 
                        'EnclosedPorch', 'BsmtUnfSF', '1stFlrSF', 
                        'BsmtFullBath', 'GarageCars', 'KitchenAbvGr', 
                        'GarageArea', 'ScreenPorch', 'MoSold', 
                        'MiscVal', 'SalePrice', 'BsmtHalfBath', 
                        '2ndFlrSF', 'YearBuilt', 'BsmtFinSF2', 
                        'OverallCond', 'GarageYrBlt', 'BsmtFinSF1', 
                        'TotalBsmtSF', 'GrLivArea', 'Fireplaces',
                        'MasVnrArea', 'WoodDeckSF', 'LotArea', 
                        'LowQualFinSF']

print('Categorical Features:',categorical_features)

print('Numerical Features:',numerical_features)
                 
for cat in categorical_features:
    data[cat].fillna(value='not-declared', inplace=True)
    print(cat,":", data[cat].unique())
    print("number of ",cat," options:", len(data[cat].unique()))

for cat in numerical_features:
    data[cat].fillna(value=0, inplace=True)
    

label_encs = dict()    
for cat in categorical_features:
    label_encs[cat] = preprocessing.LabelEncoder()
    label_encs[cat].fit(data[cat])
    new_col = label_encs[cat].transform(data[cat])
    data.loc[:, cat + '_new'] = pd.Series(new_col, index=data.index)
    
one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)

transformed_cat = one_hot_encoder.fit_transform(
        data[[x+'_new' for x in categorical_features]])
        
labels = [list(label_encs[cat].classes_) for cat in categorical_features]
new_labels = [[categorical_features[k]+labels[k][m] for m in range(len(labels[k]))] for k in range(len(categorical_features))]

lab_tot = list()
for k in range(len(labels)):
    lab_tot = lab_tot + new_labels[k]
    
X = pd.DataFrame(transformed_cat,columns=lab_tot)

for cat in numerical_features:
    X.loc[:,cat] = pd.Series(data[cat],index=data.index)
    
Y = X['SalePrice']
X.drop(['SalePrice'],axis='columns',inplace=True)

print(X.describe())

from sklearn.ensemble.forest import RandomForestRegressor

forest = RandomForestRegressor(n_estimators= 100)

forest.fit(X,Y)
print("The price accuracy is :", forest.score(X,Y)," out of 1.0")

# The output I am seeing is 0.974564107019

print("Now we do the same procedure to the test data")

numerical_features.remove('SalePrice')

for cat in categorical_features:
    test[cat].fillna(value='not-declared', inplace=True)
for cat in numerical_features:
    test[cat].fillna(value=0, inplace=True)

for cat in categorical_features:
    new_col = label_encs[cat].transform(test[cat])
    test.loc[:, cat + '_new'] = pd.Series(new_col, index=test.index)

transformed_test = one_hot_encoder.fit_transform(
        test[[x+'_new' for x in categorical_features]])
        
X_test = pd.DataFrame(transformed_test,columns=lab_tot)
for cat in numerical_features:
    X_test.loc[:,cat] = pd.Series(test[cat],index=test.index)

predictions = forest.predict(X_test)

for k in range(len(predictions)):
    print(k+1461,",",predictions[k])

# This is going to look like a dumb question, but this is my first kernel and
# the second time I have used ML algorithms. How do I output this as a submission?