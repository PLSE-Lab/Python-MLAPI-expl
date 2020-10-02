# imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# load data 
train = pd.read_csv('../input/train.csv').drop(['Id'],axis=1)
test = pd.read_csv('../input/test.csv').drop(['Id'],axis=1)

# missing value
cols_with_missing = [col for col in train.columns if train[col].isnull().sum() > 0.2*len(train)]

# remove cols
train2 = train.drop(cols_with_missing + ['SalePrice'], axis=1)
test2 = test.drop(cols_with_missing, axis=1)

df = pd.concat([train2,test2],axis=0,sort=False)

categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']

# impute missing values
num_imputer = SimpleImputer(strategy = 'median')
cat_imputer = SimpleImputer(strategy = 'most_frequent')

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# get dummies
dummies = pd.get_dummies(df[categorical_cols])

# define training data
x = pd.concat([df.select_dtypes(exclude='object').iloc[:len(train2)],dummies.iloc[:len(train2)]],axis=1)
x_test = pd.concat([df.select_dtypes(exclude='object').iloc[len(train2):],dummies.iloc[len(train2):]],axis=1)
y = np.log(train['SalePrice'])

# placeholders 
lgbm_oof = np.zeros(len(train)) 
lgbm_test = np.zeros(len(test))

rfg_oof = np.zeros(len(train))
rfg_test = np.zeros(len(test))

xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))

svr_oof = np.zeros(len(train))
svr_test = np.zeros(len(test))
                    
lasso_oof = np.zeros(len(train))
lasso_test = np.zeros(len(test))

# lgbm params
lgbm_params = {
'learning_rate': 0.01, 
'max_depth': 7, 
'boosting': 'gbdt', 
'objective': 'regression_l2', 
'metric': 'mae',
'num_leaves': 90,
'colsample_bytree': 0.6,
'subsample': 0.8,
'subsample_freq': 3,    
'min_child_weight': 3,
'reg_alpha': 0.5,
'reg_lambda':0.01,
'verbose': 500,
'num_rounds': 10000,
'seed': 1}

k = 25
kf = KFold(n_splits=k, random_state=42)
rs = RobustScaler()

# begin training
for i, (train_index,test_index) in enumerate(kf.split(x,y)):
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = x.iloc[train_index], x.iloc[test_index]
    
    #lgbm
    lgbm_train = lgbm.Dataset(X_train, label = y_train)
    lgbm_valid = lgbm.Dataset(X_valid, label = y_valid)
    model = lgbm.train(lgbm_params,train_set = lgbm_train, early_stopping_rounds=100, 
                                valid_sets = lgbm_valid, )
                                
    oof_pred = model.predict(X_valid)
    test_pred = model.predict(x_test)
    
    lgbm_oof[test_index] = oof_pred 
    lgbm_test += np.exp(test_pred)
    
    #rfg
    model = RandomForestRegressor(n_estimators=200, random_state=1)
    model.fit(X_train,y_train)
    
    oof_pred = model.predict(X_valid)
    test_pred = model.predict(x_test)
    
    rfg_oof[test_index] = oof_pred 
    rfg_test += np.exp(test_pred)
    
    #xgb
    model = XGBRegressor(learning_rate=0.05,min_child_weight=0, n_estimators=3460, 
                         max_depth=3, subsample = 0.8, colsample_bytree =0.6, reg_alpha=0.5, reg_lambda = 0.01, random_state=1)
    model.fit(X_train,y_train)
    
    oof_pred = model.predict(X_valid)
    test_pred = model.predict(x_test)
    
    xgb_oof[test_index] = oof_pred 
    xgb_test += np.exp(test_pred)
    
    #svr
    model = SVR(C=20, epsilon=0.008, gamma=0.0003)
    model.fit(rs.fit_transform(X_train),y_train)
    
    oof_pred = model.predict(rs.fit_transform(X_valid))
    test_pred = model.predict(rs.fit_transform(x_test))
    
    svr_oof[test_index] = oof_pred 
    svr_test += np.exp(test_pred)
    
    #lasso
    model = Lasso(alpha=0.0001,random_state=1, max_iter=2000)
    model.fit(rs.fit_transform(X_train),y_train)
    
    oof_pred = model.predict(rs.fit_transform(X_valid))
    test_pred = model.predict(rs.fit_transform(x_test))
    
    lasso_oof[test_index] = oof_pred 
    lasso_test += np.exp(test_pred)
    

# save
lgbm_test_pred = pd.DataFrame(lgbm_test)/k
rfg_test_pred = pd.DataFrame(rfg_test)/k
xgb_test_pred = pd.DataFrame(xgb_test)/k
svr_test_pred = pd.DataFrame(svr_test)/k
lasso_test_pred = pd.DataFrame(lasso_test)/k

# score
print('lgbm:', mean_absolute_error(np.exp(y),np.exp(lgbm_oof)))
print('rfg:', mean_absolute_error(np.exp(y),np.exp(rfg_oof)))
print('xgb:', mean_absolute_error(np.exp(y),np.exp(xgb_oof)))
print('svr:', mean_absolute_error(np.exp(y),np.exp(svr_oof)))
print('lasso:', mean_absolute_error(np.exp(y),np.exp(lasso_oof)))

#blend
blend = 0.4*lgbm_oof + 0.1*rfg_oof + 0.4*xgb_oof + 0.05*svr_oof + 0.05*lasso_oof
print('blend:', mean_absolute_error(np.exp(y),np.exp(blend)))
sub_blend = 0.4*lgbm_test_pred + 0.1*rfg_test_pred + 0.4*xgb_test_pred + 0.05*svr_test_pred + 0.05*lasso_test_pred

#submit
sub = pd.read_csv('../input/sample_submission.csv')
sub['SalePrice'] = sub_blend
sub.to_csv('submission_blend.csv', index=False)

