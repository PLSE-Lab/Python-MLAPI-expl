from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import lightgbm as lgb

path = '../input/'
train = pd.read_csv(path+'train.csv')
#outliers from another kernel (thanks)
train.drop([30, 88, 462, 631, 1322],inplace=True,axis=0)
train.drop('Id',inplace=True,axis=1)
target = np.log1p(train['SalePrice'])
train.drop('SalePrice',inplace=True,axis=1)

test = pd.read_csv(path+'test.csv')
test.drop('Id',inplace=True,axis=1)

submittal = pd.read_csv(path+'sample_submission.csv')

#impute buckets from another kernel (thanks)
imp_zero = ['BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath','MasVnrArea','GarageYrBlt','GarageCars','GarageArea','YearRemodAdd']
imp_none = ['BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',]
imp_mode = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr','BldgType', 'CentralAir', 'Condition1', 'Condition2', 'Electrical','EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st','Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation','FullBath', 'Functional', 'GrLivArea', 'HalfBath', 'Heating','HeatingQC', 'HouseStyle', 'KitchenAbvGr', 'KitchenQual','LandContour', 'LandSlope', 'LotArea', 'LotConfig', 'LotFrontage','LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning', 'MasVnrType','MiscFeature', 'MiscVal', 'MoSold', 'Neighborhood', 'OpenPorchSF','OverallCond', 'OverallQual', 'PavedDrive', 'PoolArea', 'PoolQC','RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType','ScreenPorch', 'Street', 'TotRmsAbvGrd', 'Utilities', 'WoodDeckSF','YearBuilt', 'YrSold']
train = train.loc[:,imp_zero+imp_none+imp_mode]
num_cols = list(train.dtypes.reset_index()[train.dtypes.reset_index(drop=True) != 'object'].index)
obj_cols = list(train.dtypes.reset_index()[train.dtypes.reset_index(drop=True) == 'object'].index)

class fillfunc():
    def zero(X) :
        return X.fillna(0)
    def none(X) :
        return X.fillna('none')
    def mode(X) :
        return X.fillna(X.mode().iloc[0])

class ProjectImputer(BaseEstimator,TransformerMixin) :
    def __init__(self,func,cols) :
        self.func = func
        self.cols = cols
    def _getcols(self,X) :
        return X[self.cols]
    def fit(self,X,y=None) :
        return self
    def transform(self,X,y=None) :
        X = self._getcols(X)
        return self.func(X)
    
class ProjectScaler(BaseEstimator,TransformerMixin) :
    def __init__(self,func,cols):
        self.func = func
        self.cols = cols
    def _getcols(self,X) :
        return X[:,self.cols]
    def fit(self,X,y=None) :
        X = self._getcols(X)
        self.func.fit(X,y)
        return self
    def transform(self,X) :
        X = self._getcols(X)
        return self.func.transform(X)

class LGBRegressorCV(BaseEstimator,RegressorMixin) :
    def __init__(self,fit_params=None,n_splits=5) :
        self.fit_params = fit_params
        self.n_splits = n_splits
    def fit(self,X,y) : 
        oof_preds = np.zeros(X.shape[0]) 
        self.M = []
        X = pd.DataFrame(X)
        folds = KFold(n_splits= self.n_splits, shuffle=True,random_state=2077)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            dtrain = lgb.Dataset(data=X.iloc[train_idx], 
                                 label=y.iloc[train_idx], 
                                 )
            dvalid = lgb.Dataset(data=X.iloc[valid_idx], 
                                 label=y.iloc[valid_idx], 
                                 )
            m = lgb.train(
                train_set=dtrain,
                valid_sets=[dtrain, dvalid],
                params=self.fit_params,
                num_boost_round=10000,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            oof_preds[valid_idx] = m.predict(X.iloc[valid_idx])
            self.M.append(m)
            print( 'fold %2d > RMSLE : FIT %.4f CV %.4f' %  ( n_fold+1, np.sqrt(mean_squared_error(y.iloc[train_idx],m.predict(X.iloc[train_idx]))) ,np.sqrt(mean_squared_error(y.iloc[valid_idx],oof_preds[valid_idx])) ) ) 
        print( 'Final CV RMSLE : %.4f' % (np.sqrt(mean_squared_error(y, oof_preds)) ) )
        return self
    def predict(self,X) :
        sub_preds = []
        for m in self.M :
            sub_preds.append(m.predict(X))
        return np.mean(sub_preds,axis=0)
    
def main() :
    fit_params = {
        'boosting_type': 'gbrt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1,
        'subsample' : 2/3, 
        'subsample_freq' : 5, 
        'sub_feature' : 2/3,
        'max_bin' : 50,
        'num_leaves' : 6,
        'min_data_in_leaf' : 6, 
        'lambda_l1' : 0,
        'lambda_l2' : 0,
        'nthread': -1,
        'verbose': -1,
    }
    pipe = Pipeline([
        ('uni_imp',FeatureUnion([
            ('imp_zero',ProjectImputer(fillfunc.zero,imp_zero)),
            ('imp_none',ProjectImputer(fillfunc.none,imp_none)),
            ('imp_mode',ProjectImputer(fillfunc.mode,imp_mode)),
        ])),
        ('uni_scl',FeatureUnion([
            ('scl-num', ProjectScaler(StandardScaler(),num_cols)),
            ('scl-obj', ProjectScaler(OneHotEncoder(sparse=False),obj_cols)),       
        ])),
        ('lgb', LGBRegressorCV(fit_params=fit_params))
    ])
    pipe.fit(train,target)

    sub = pipe.predict(test)
    sub = np.expm1(sub)
    sub = np.round(sub,-3)
    submittal['SalePrice'] = sub
    submittal.to_csv('submittal.csv',index=None)

if __name__ == '__main__':
    main()