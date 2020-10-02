
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import scipy.stats as st
from math import sqrt
import os

from six.moves import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation, metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#data_root = '../input' # Change me to store data elsewhere

#reading the train and test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#--------------------------------------------------
trainSize = train.shape #size of the train dataset
testSize = test.shape #size of the test dataset
print("train dataset size = %s x %s" % (trainSize))
print("test dataset size = %s x %s" % (testSize))

y = train.SalePrice
y = np.log1p(y)
train.drop('SalePrice',1,inplace=True)
all_data = pd.concat([train,test]) 

##drop attributes with many missing values
#too many missing values
all_data.drop("Alley", 1,inplace=True)
all_data.drop("Fence", 1,inplace=True)
all_data.drop("MiscFeature", 1,inplace=True)
all_data.drop("PoolQC", 1,inplace=True)
all_data.drop("FireplaceQu", 1,inplace=True)
#non-intuitive features
all_data.drop("GarageArea", 1,inplace=True)
all_data.drop("MSSubClass", 1,inplace=True)
all_data.drop("GarageYrBlt", 1,inplace=True)
all_data.drop("RoofMatl", 1,inplace=True)
#convert year to age
Yr = max(all_data['YearBuilt'])
all_data['BuildingAge'] = all_data['YearBuilt'].apply(lambda x: Yr-x if not pd.isnull(x) else 'None')
all_data['RemodelAge'] = all_data['YearRemodAdd'].apply(lambda x: Yr-x if not pd.isnull(x) else 'None')
all_data['SellAge'] = all_data['YrSold'].apply(lambda x: Yr-x if not pd.isnull(x) else 'None')
#drop old variables
all_data.drop("YearBuilt", 1,inplace=True);
all_data.drop("RemodelAge", 1,inplace=True);
all_data.drop("YrSold", 1,inplace=True);

#extraxt train and test datasets
train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]

#divide features into numeric and categorical 
numeric = [c for c in train.columns if train.dtypes[c] != 'object']
numeric.remove('Id')
print("Number of Numeric Attributes = %s" % (len(numeric)))
categorical = [c for c in train.columns if train.dtypes[c] == 'object']
print("Number of Categorical Attributes = %s" % (len(categorical)))

##preprocessing of numeric features
feature = train[numeric].dropna()
skewed_feats = feature.apply(skew) #compute skewness
skewed_feats = skewed_feats[(skewed_feats > 0.5) | (skewed_feats < -0.5)]
skewed_feats = skewed_feats.index

#log transformation to resolve skewness on train and test data
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

pd.options.mode.chained_assignment = None  # default='warn'
#standard scale (zero mean and unit variance)
all_data[numeric] = all_data[numeric].fillna(all_data[numeric].mean())
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
columns=x_train.columns
x_train[numeric] = StandardScaler().fit_transform(x_train[numeric])
x_train = pd.DataFrame(x_train, index=x_train.index, columns=x_train.columns)
x_test[numeric] = StandardScaler().fit_transform(x_test[numeric])
x_test = pd.DataFrame(x_test, index=x_test.index, columns=x_test.columns)
all_data = pd.concat([x_train,x_test])

##preprocessing of categorical features
feature = all_data[categorical]
#identify the most frequent level of each categorical feature
frequentLevel = feature.apply(lambda x: x.value_counts().idxmax())
def itemReplace (column,value):
    frequentLevel[column] = value
itemReplace('BsmtFinSF1','None')
itemReplace('BsmtFinSF2','None')
itemReplace('GarageType','None')
itemReplace('GarageFinish','None')
itemReplace('GarageQual','None')
itemReplace('GarageCond','None')
all_data[categorical] = feature.fillna(frequentLevel)

#transform categorical variables into dummy variables
all_data = pd.get_dummies(all_data)

x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
columns=x_train.columns
print(x_train.shape)
print(x_test.shape)

var = 'GrLivArea'
#deleting points
x_train.sort_values(by = var, ascending = False)[:2]
x_train = x_train.drop(x_train.index[[1298,523]])
y = y.drop(y.index[[1298,523]])

var = 'TotalBsmtSF'
def correct(dataset,var):
    idx = columns.get_loc(var)
    n = dataset.shape[0]
    for i in range(n):
        if  dataset.iloc[i,idx] < -4:
            dataset.iloc[i,idx]=0
    return dataset
           
x_train = correct(x_train,var)
x_test = correct(x_test,var)

#deleting points
x_train.sort_values(by = var, ascending = True)[:1]
x_train = x_train.drop(x_train.index[[871]])
y = y.drop(y.index[[871]])

x_train.drop("Id", 1,inplace=True)
x_test.drop("Id", 1,inplace=True)
names = list(x_train)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y)
ntrain = x_train.shape[0]
ntest = x_test.shape[0]

NFOLDS = 4
SEED = 0
NROWS = None

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
    
    
def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et_params = {
    'n_jobs': 16,
    'n_estimators': 20,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 20,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.9,
    'silent': 1,
    'subsample': 0.9,
    'learning_rate': 0.15,
    'objective': 'reg:linear',
    'max_depth': 3,
    'num_parallel_tree': 1,
    'min_child_weight': 9,
    'gamma': 0.15,
    'eval_metric': 'rmse',
    'n_estimators': 65,
    'nrounds': 500
}


rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.0005
}

xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)
print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}


res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=5, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv('../input/sample_submission.csv')
submission.iloc[:, 1] = gbdt.predict(dtest)
saleprice = np.expm1(submission['SalePrice'])
submission['SalePrice'] = saleprice
submission.to_csv('xgstacker_starter.csv', index=None)



