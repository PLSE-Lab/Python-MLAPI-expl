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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

test_id = test_df['Id']
Y_train = train_df['SalePrice']
Y_train = np.log1p(Y_train)
train_df.drop(['SalePrice'], inplace=True, axis=1)

df = train_df.append(test_df, ignore_index=True)
df.drop(['Id'], inplace=True, axis=1)
X = df
X['Alley'].fillna('NA', inplace=True)
X['PoolQC'].fillna('NP', inplace=True)
X['Fence'].fillna('NF', inplace=True)
X['MasVnrType'].fillna('None', inplace=True)
X['BsmtQual'].fillna('NB', inplace=True)
X['BsmtCond'].fillna('NB', inplace=True)
X['BsmtExposure'].fillna('NB', inplace=True)
X['BsmtFinType1'].fillna('NB', inplace=True)
X['BsmtFinType2'].fillna('NB', inplace=True)
X['Electrical'].fillna('SBrkr', inplace=True)
X['FireplaceQu'].fillna('NF', inplace=True)
X['GarageType'].fillna('NG', inplace=True)
X['GarageYrBlt'].fillna(0, inplace=True)
X['GarageFinish'].fillna('NG', inplace=True)
X['GarageQual'].fillna('NG', inplace=True)
X['GarageCond'].fillna('NG', inplace=True)
X['MSZoning'].fillna('RL', inplace=True)
X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)
X['MasVnrArea'].fillna(0, inplace=True)
X['Utilities'].fillna('AllPub', inplace=True)
X['Exterior1st'].fillna('VinylSd', inplace=True)
X['Exterior2nd'].fillna('VinylSd', inplace=True)
X['BsmtFinSF1'].fillna(X['BsmtFinSF1'].mean(), inplace=True)
X['BsmtFinSF2'].fillna(X['BsmtFinSF2'].mean(), inplace=True)
X['BsmtUnfSF'].fillna(X['BsmtUnfSF'].mean(), inplace=True)
X['TotalBsmtSF'].fillna(X['TotalBsmtSF'].mean(), inplace=True)
X['BsmtFullBath'].fillna(X['BsmtFullBath'].mean(), inplace=True)
X['BsmtHalfBath'].fillna(X['BsmtHalfBath'].mean(), inplace=True)
X['KitchenQual'].fillna('TA', inplace=True)
X['Functional'].fillna('Typ', inplace=True)
X['GarageCars'].fillna(2, inplace=True)
X['GarageArea'].fillna(X['GarageArea'].mean(), inplace=True)
X['SaleType'].fillna('WD', inplace=True)
X.drop(['MiscFeature'], inplace=True, axis=1)

# label and one-hot encode data
cat_cols = X.dtypes.index[X.dtypes == 'object']
cont_cols = X.dtypes.index[X.dtypes != 'object']
for column in cat_cols:

	le = LabelEncoder()
	X[column] = le.fit_transform(X[column])

	one_hot = pd.get_dummies(X[column], prefix=column)
	X.drop(column, axis=1, inplace=True)
	X = X.join(one_hot)

# replace numerical columns
X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)
X['MasVnrArea'].fillna(0, inplace=True)

# Skew correct numerical columns
X['MSSubClass'] = np.log1p(X['MSSubClass'])
X['LotFrontage'] = np.log1p(X['LotFrontage'])
X['LotArea'] = np.log1p(X['LotArea'])
X['OverallCond'] = np.log1p(X['OverallCond'])
X['MasVnrArea'] = np.log1p(X['MasVnrArea'])
X['BsmtFinSF1'] = np.log1p(X['BsmtFinSF1'])
X['BsmtFinSF2'] = np.log1p(X['BsmtFinSF2'])
X['BsmtUnfSF'] = np.log1p(X['BsmtUnfSF'])
X['TotalBsmtSF'] = np.log1p(X['TotalBsmtSF'])
X['1stFlrSF'] = np.log1p(X['1stFlrSF'])
X['2ndFlrSF'] = np.log1p(X['2ndFlrSF'])
X['GrLivArea'] = np.log1p(X['GrLivArea'])
X['WoodDeckSF'] = np.log1p(X['WoodDeckSF'])
X['OpenPorchSF'] = np.log1p(X['OpenPorchSF'])
X['EnclosedPorch'] = np.log1p(X['EnclosedPorch'])
X['ScreenPorch'] = np.log1p(X['ScreenPorch'])
X['MiscVal'] = np.log1p(X['MiscVal'])

X_train = X[:train_df.shape[0]]

print('Doing search grid cv using Ridge...')
seed = 7
alpha = [1, 5, 8, 10, 20, 30]
param_grid = {'alpha': alpha}
model = Ridge()
kfold = KFold(n_splits=5, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold,
		n_jobs=-1, scoring='mean_squared_error')
grid_result = grid.fit(X_train, Y_train)

print('Best: %f using %s'%(np.sqrt(-grid_result.best_score_),
	grid_result.best_params_))
	
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, parm in zip(means, stds, params):
	mean = np.sqrt(-mean)
	std = np.sqrt(std)
	print('%f (%f) with: %r'%(mean, std, parm))


predicted = grid.predict(X_train)
X_test = X[train_df.shape[0]:]
test_pred = np.expm1(grid.predict(X_test))

submission = pd.DataFrame(test_pred, index=test_id.values, columns=['SalePrice'])
submission.index.name = 'Id'
submission.to_csv('submission.csv')

# TODO - for improvement
# find and remove correlated features
# normalise features to same scale
# Try ensemble
# Try XGBoost