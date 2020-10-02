import pandas
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn import ensemble

#load data
traindata = pandas.read_csv("../input/train.csv")
testdata = pandas.read_csv("../input/test.csv")
trainsize = traindata.shape[0]
housedata = pandas.concat((traindata.loc[:,'MSSubClass':'SaleCondition'],
                          testdata.loc[:,'MSSubClass':'SaleCondition']))
y = traindata['SalePrice']

#data cleasing
housedata['LotFrontage']= housedata['LotFrontage'].fillna(housedata['LotFrontage'].mean())
housedata['Alley'] = housedata['Alley'].fillna('No access')
numLotShape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
housedata['LotShape'] = housedata['LotShape'].replace(numLotShape).fillna(0)
numLandContour = {'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3}
housedata['LandContour'] = housedata['LandContour'].replace(numLandContour).fillna(0)
numLandSlope = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
housedata['LandSlope'] = housedata['LandSlope'].replace(numLandSlope).fillna(0)
housedata['MasVnrArea'].fillna(housedata['MasVnrArea'].mean())
numExterQual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['ExterQual'] = housedata['ExterQual'].replace(numExterQual).fillna(0)
numExterCond = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['ExterCond'] = housedata['ExterCond'].replace(numExterCond).fillna(0)
numBsmtQual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['BsmtQual'] = housedata['BsmtQual'].replace(numBsmtQual).fillna(0)
numBsmtCond = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['BsmtCond'] = housedata['BsmtCond'].replace(numBsmtCond).fillna(0)
numBsmtExposure = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
housedata['BsmtExposure'] = housedata['BsmtExposure'].replace(numBsmtExposure).fillna(0)
numBsmtFinType = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
housedata['BsmtFinType1'] = housedata['BsmtFinType1'].replace(numBsmtFinType).fillna(0)
housedata['BsmtFinType2'] = housedata['BsmtFinType2'].replace(numBsmtFinType).fillna(0)
numHeatingQC = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['HeatingQC'] = housedata['HeatingQC'].replace(numHeatingQC).fillna(0)
numKitchenQual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['KitchenQual'] = housedata['KitchenQual'].replace(numKitchenQual).fillna(0)
numFireplaceQu = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['FireplaceQu'] = housedata['FireplaceQu'].replace(numFireplaceQu).fillna(0)
numGarageQual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['GarageQual'] = housedata['GarageQual'].replace(numGarageQual).fillna(0)
numGarageCond = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
housedata['GarageCond'] = housedata['GarageCond'].replace(numGarageCond).fillna(0)
numPoolQC = {'NA': 0,'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
housedata['PoolQC'] = housedata['PoolQC'].replace(numPoolQC).fillna(0)
housedata['YearBuilt'] = 2017 - housedata['YearBuilt']
housedata['YearRemodAdd'] = 2017 - housedata['YearRemodAdd']
housedata['GarageYrBlt'] = 2017 - housedata['GarageYrBlt']
housedata['YrSold'] = 2017 - housedata['YrSold']

numeric_feats = housedata.dtypes[housedata.dtypes != "object"].index

#generate easy categories
housedata = pandas.get_dummies(housedata)
housedata = housedata.fillna(0)

#log transform skewed numeric features:
skewness = housedata[numeric_feats].apply(lambda x: skew(x))
left_skewed_feats = skewness[skewness > 0.5].index
right_skewed_feats = skewness[skewness < -0.5].index
housedata[left_skewed_feats] = np.log1p(housedata[left_skewed_feats])
housedata[right_skewed_feats] = np.exp(housedata[right_skewed_feats])
housedata[numeric_feats] = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(housedata[numeric_feats])

Xtrain = housedata[:trainsize]
Xtest = housedata[trainsize:]
y = np.log1p(y)

# Select best parameter for linear regression model



alphas = np.linspace(0.0008, 0.0011, num=11)
scores = [
     np.sqrt(-cross_val_score(ElasticNet(alpha), Xtrain, y, scoring="neg_mean_squared_error", cv=5)).mean()
     for alpha in alphas
]
alpha1 = alphas[np.argmin(scores)]


alphas = np.linspace(0.0002, 0.0005, num=11)
scores = [
     np.sqrt(-cross_val_score(Lasso(alpha), Xtrain, y, scoring="neg_mean_squared_error")).mean()
     for alpha in alphas
]
alpha2 = alphas[np.argmin(scores)]

#params = {'n_estimators': 2000, 'max_depth': 5, 'learning_rate': 0.005, 'loss': 'ls'}
#clf = ensemble.GradientBoostingRegressor(**params)
#clf.fit(Xtrain, y)


linear_model1 = ElasticNet(alpha1)
linear_model1.fit(Xtrain, y)
linear_model2 = Lasso(alpha2)
linear_model2.fit(Xtrain, y)
testdata['SalePrice'] = np.expm1(linear_model2.predict(Xtest))
#testdata['SalePrice'] = np.expm1((linear_model1.predict(Xtest)+linear_model2.predict(Xtest))/2)
#testdata['SalePrice'] = np.expm1((linear_model1.predict(Xtest)+clf.predict(Xtest))/2)
testdata.to_csv('submission.csv', index=False, columns=['Id', 'SalePrice'])