import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import lightgbm as ltb

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

X = pd.concat([train_data.iloc[:,1:-1],test_data.iloc[:,1:]]) #features = everything except SalePrice and Id 

#numbertype that is category
X['MSSubClass'] = X['MSSubClass'].astype("object").fillna('0')
X['MoSold'] = X['MoSold'].astype("object").fillna('unknown')

#category that describes a qualitative value 
X['Utilities'] = X['Utilities'].map({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0})
X['LandSlope'] = X['LandSlope'].map({'Gtl': 2, 'Mod': 1, 'Sev': 0})
X['ExterQual'] = X['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
X['ExterCond'] = X['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
X['BsmtQual'] = X['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA': 0})
X['BsmtCond'] = X['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA': 0})
X['BsmtExposure'] = X['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
X['BsmtFinType1'] = X['BsmtFinType1'].map({'GLQ': 6, 'GLQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2,'Unf': 1, 'NA': 0})
X['BsmtFinType2'] = X['BsmtFinType2'].map({'GLQ': 6, 'GLQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2,'Unf': 1, 'NA': 0})
X['HeatingQC'] = X['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
X['CentralAir'] = X['CentralAir'].map({'Y': 1,'N': 0})
X['Electrical'] = X['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
X['KitchenQual'] = X['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
X['Functional'] = X['Functional'].map({ 'Typ' : 7, 'Min1' : 6, 'Min2' : 5, 'Mod': 4, 'Maj1' : 3, 'Maj2' : 2, 'Sev' : 1, 'Sal' : 0})
X['FireplaceQu'] = X['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA': 0})
X['GarageType'] = X['GarageType'].map({'2Types': 6, 'Attchd': 5, 'Basment': 4, 'BuiltIn': 3, 'CarPort': 2, 'Detchd': 1, 'NA': 0})
X['GarageFinish'] = X['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1,'NA': 0})
X['GarageQual'] = X['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA': 0})
X['GarageCond'] = X['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA': 0})
X['PavedDrive'] = X['PavedDrive'].map({'Y': 2, 'P': 1,'N': 0})
X['PoolQC'] = X['PoolQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'NA': 1})
X['Fence'] = X['Fence'].map({'GdPrv': 5, 'GdWo': 4, 'MnPrv': 3, 'MnWw': 2, 'NA': 1})

#false NaN
X['Alley'] = X['Alley'].fillna('no_alley')

#new features
X['Fireplace&Qu'] = X['Fireplaces'] * X['FireplaceQu']
X['AgeAtSale'] = X['YrSold'] - X['YearRemodAdd']
X['OverallQual&Cond'] = X['OverallQual'] * X['OverallCond']
X['GarageQual&Cond'] = X['GarageQual'] * X['GarageCond']
X['ExterQual&Cond'] = X['ExterQual'] * X['ExterCond']
X['KitchenQual&AbvGr'] = X['KitchenAbvGr'] * X['KitchenQual']
X['GarageArea&Qual'] = X['GarageArea'] * X['GarageQual']
X['PoolArea&QC'] = X['PoolArea'] * X['PoolQC']
X['TotalBath'] = X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']) + X['FullBath'] + (0.5 * X['HalfBath'])
X['TotalFlrsSF'] = X['1stFlrSF'] + X['2ndFlrSF']
X['TotalSF'] = X['GrLivArea'] + X['TotalBsmtSF']
X['TotalPorchSF'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
X['LowQualSFPercentage'] = X['LowQualFinSF'] / X['TotalFlrsSF']

X = pd.get_dummies(X,dummy_na=True)
X = X.loc[:, (X != 0).any(axis=0)] #remove columns that are completely 0 
X = X.fillna(X.mean())

#print(X.iloc[0:1460].corrwith(train_data.iloc[:,-1]).fillna(0).abs().sort_values())

#split X into training and test data
y_train = train_data.iloc[:,-1] #label = SalePrice
X_train = X.iloc[0:1460]
X_test = X.iloc[1460:2919]

#scale data before regression
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

r1 = GradientBoostingRegressor(random_state=0,n_estimators=400,subsample=0.7)
r2 = HuberRegressor(epsilon=1.65)
r3 = ElasticNet(alpha=0.009,random_state=0,l1_ratio=0.9,max_iter=1000)
r4 = ExtraTreesRegressor(n_estimators=400,random_state=0,min_samples_split=3)
r5 = KNeighborsRegressor(n_neighbors=2,p=1)
r6 = ltb.LGBMRegressor(learning_rate=0.3,boosting_type='dart',random_state=0,n_estimators=400)
r7 = PassiveAggressiveRegressor(random_state=0)

vreg = VotingRegressor([('grb', r1), ('hub', r2),('ela',r3),('ext',r4),('kne',r5),('lgb',r6),('par',r7)],weights=[1,0.55,0.05,0.23,0.04,0.25,0.05])

#score = cross_val_score(vreg, X_train, y_train,cv=5,scoring='neg_mean_squared_log_error')
#print(np.sqrt(-score.mean()))
#print(np.sqrt(-score.min()))

y_pred = vreg.fit(X_train, y_train).predict(X_test)

submission = pd.DataFrame()
submission['Id'] = test_data['Id']
submission['SalePrice'] = y_pred
submission.to_csv('submission.csv', header=True, index=False)