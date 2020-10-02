import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def transform_cat(df):
    cat_columns = df.select_dtypes(['category']).columns
    print(cat_columns)
    for col in cat_columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].apply(lambda x: x.cat.codes)


train = pd.read_csv('../input/train.csv')
# transform_cat(train)
test = pd.read_csv('../input/test.csv')
train_id = train['Id']
test_id = test['Id']
all = pd.concat([train, test])
result = pd.get_dummies(all)
train = result.loc[result['Id'].isin(train_id)]
test = result.loc[result['Id'].isin(test_id)]
# X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice',axis=1), train['SalePrice'], test_size=0.2)

def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))


def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)
    test[feature] = np.log1p(test[feature].values)


def quadratic(feature):
    train[feature + '2'] = train[feature] ** 2
    test[feature + '2'] = test[feature] ** 2


log_transform('GrLivArea')
log_transform('1stFlrSF')
log_transform('2ndFlrSF')
log_transform('TotalBsmtSF')
log_transform('LotArea')
log_transform('LotFrontage')
log_transform('KitchenAbvGr')
log_transform('GarageArea')

# boolean = ['TotalBsmtSF', 'GarageArea', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF',
#            'OpenPorchSF', 'PoolArea', 'YearBuilt']
# quantitative=['GrLivArea','1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','LotFrontage','KitchenAbvGr','GarageArea']

#X_train, X_test, y_train, y_test = train_test_split(train.drop(['SalePrice', 'Id'], axis=1), train['SalePrice'], test_size=0.2)
# features = quantitative + boolean + qdr
lasso = GradientBoostingRegressor()
# tr= train.drop(['SalePrice','Id'],axis=1)
tr = train.drop(['SalePrice','Id'],axis=1)
X = tr.fillna(0.).values
Y = train['SalePrice'].values
lasso.fit(X, np.log(Y))
ids = test['Id']
test=test.drop(['Id','SalePrice'],axis=1)
Ypred = np.exp(lasso.predict(test.fillna(0.).values))
sub = pd.DataFrame({
    "Id": ids,
    "SalePrice": Ypred
})
sub.to_csv("prices_submission.csv", index=False)
